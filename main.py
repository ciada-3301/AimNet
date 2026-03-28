"""
AimNet — Supervised Learning + Ballistic Visualizer
====================================================
Trains a neural network to predict the optimal firing angle for a projectile,
then opens an interactive targeting terminal where you can drag firing and
target positions and watch the AI fire its shot in real time.
"""

import math
import time
import threading
import tkinter as tk
from tkinter import font as tkfont

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ── Physics ────────────────────────────────────────────────────────────────

G   = 9.8
V   = 40.0   # muzzle velocity (m/s)

def analytical_theta(xt, yt, v=V, g=G):
    a = (g * xt**2) / (2 * v**2)
    b = -xt
    c = yt + (g * xt**2) / (2 * v**2)
    disc = b**2 - 4*a*c
    if disc < 0:
        return None
    sqrt_d = math.sqrt(disc)
    candidates = [math.atan((-b + s*sqrt_d)/(2*a)) for s in (1,-1) if (-b + s*sqrt_d)/(2*a) > 0]
    return min(candidates) if candidates else None

def simulate_trajectory(x0, y0, theta, v=V, g=G, dt=0.01):
    """Returns list of (x,y) points until projectile hits ground."""
    pts = []
    t = 0
    while True:
        x = x0 + v*math.cos(theta)*t
        y = y0 + v*math.sin(theta)*t - 0.5*g*t**2
        pts.append((x, y))
        if y < y0 and t > 0.05:
            break
        t += dt
    return pts

def simulate_miss(xt, yt, theta, x0=0, y0=0, v=V, g=G, dt=0.005):
    min_dist = float('inf')
    t = 0
    while True:
        x = x0 + v*math.cos(theta)*t
        y = y0 + v*math.sin(theta)*t - 0.5*g*t**2
        if y < y0 and t > 0.05:
            break
        dist = math.sqrt((x-xt)**2 + (y-yt)**2)
        if dist < min_dist:
            min_dist = dist
        if dist < 0.5:
            return 0.0
        t += dt
    return min_dist

# ── Model ──────────────────────────────────────────────────────────────────

class Aimnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mu_head = nn.Linear(256, 1)

    def forward(self, x):
        return torch.sigmoid(self.mu_head(self.net(x))) * (math.pi / 2)

def make_state(xt, yt, v=V):
    return torch.tensor([xt/100.0, yt/30.0, v/50.0], dtype=torch.float32)

# ── Training ───────────────────────────────────────────────────────────────

def train(steps=8000, batch=200, on_progress=None, on_done=None):
    model     = Aimnet()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    for step in range(steps + 1):
        states, targets = [], []
        while len(states) < batch:
            xt = torch.rand(1).item() * 90 + 10
            yt = torch.rand(1).item() * 28
            theta_opt = analytical_theta(xt, yt)
            if theta_opt is not None:
                states.append(make_state(xt, yt))
                targets.append(theta_opt)

        states_t  = torch.stack(states)
        targets_t = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)

        pred = model(states_t)
        loss = F.mse_loss(pred, targets_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 200 == 0:
            with torch.no_grad():
                err_deg = (pred - targets_t).abs().mean().item() * (180/math.pi)
            if on_progress:
                on_progress(step, steps, loss.item(), err_deg)

    if on_done:
        on_done(model)

# ── Visualizer ─────────────────────────────────────────────────────────────

PHOSPHOR  = "#00ff41"
PHOSPHOR2 = "#00cc33"
PHOSPHOR3 = "#007a1f"
DIM       = "#004010"
BG        = "#000a00"
BG2       = "#010f01"
AMBER     = "#ffb830"
RED       = "#ff3333"
GRID_COL  = "#001f00"

class BallisticTerminal:
    CANVAS_W = 780
    CANVAS_H = 420
    PAD_L    = 60
    PAD_R    = 40
    PAD_T    = 40
    PAD_B    = 60
    WORLD_W  = 130   # metres shown
    WORLD_H  = 60

    def __init__(self, root, model):
        self.root  = root
        self.model = model
        model.eval()

        self.cx0   = 10.0   # cannon x (metres)
        self.cy0   = 0.0    # cannon y (metres) — always ground
        self.tx    = 80.0   # target x
        self.ty    = 15.0   # target y

        self.anim_pts    = []
        self.anim_idx    = 0
        self.anim_id     = None
        self.hit         = False
        self.fired_theta = None
        self.true_theta  = None

        self._drag = None   # which thing is being dragged: 'cannon' or 'target'

        self._build_ui()
        self._draw_scene()

    # ── layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        root.title("AimNet — Ballistic Targeting Terminal")
        root.configure(bg=BG)
        root.resizable(False, False)

        try:
            mono   = tkfont.Font(family="Courier New", size=10, weight="bold")
            mono_s = tkfont.Font(family="Courier New", size=8)
            mono_l = tkfont.Font(family="Courier New", size=13, weight="bold")
            mono_xl= tkfont.Font(family="Courier New", size=16, weight="bold")
        except Exception:
            mono   = tkfont.Font(size=10, weight="bold")
            mono_s = tkfont.Font(size=8)
            mono_l = tkfont.Font(size=13, weight="bold")
            mono_xl= tkfont.Font(size=16, weight="bold")

        self._fonts = dict(mono=mono, mono_s=mono_s, mono_l=mono_l, mono_xl=mono_xl)

        # ── top bar ──
        top = tk.Frame(root, bg=BG, pady=6)
        top.pack(fill="x", padx=16)

        tk.Label(top, text="▶  AIMNET BALLISTIC TARGETING SYSTEM  ◀",
                 bg=BG, fg=PHOSPHOR, font=mono_xl).pack(side="left")
        tk.Label(top, text="MODEL: TRAINED  |  v=40m/s  |  g=9.8m/s²",
                 bg=BG, fg=PHOSPHOR3, font=mono_s).pack(side="right")

        # ── separator ──
        tk.Frame(root, bg=PHOSPHOR3, height=1).pack(fill="x", padx=16)

        # ── main row ──
        main = tk.Frame(root, bg=BG)
        main.pack(fill="both", expand=True, padx=16, pady=10)

        # canvas
        self.canvas = tk.Canvas(
            main,
            width=self.CANVAS_W, height=self.CANVAS_H,
            bg=BG2, highlightthickness=1, highlightbackground=PHOSPHOR3,
            cursor="crosshair"
        )
        self.canvas.pack(side="left")
        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # right sidebar
        side = tk.Frame(main, bg=BG, width=230)
        side.pack(side="left", fill="y", padx=(14,0))
        side.pack_propagate(False)

        def sep():
            tk.Frame(side, bg=PHOSPHOR3, height=1).pack(fill="x", pady=8)

        tk.Label(side, text="[ TARGETING DATA ]", bg=BG, fg=PHOSPHOR2,
                 font=mono, anchor="w").pack(fill="x")
        sep()

        # coord rows
        self._lbl_cannon = self._make_data_row(side, "CANNON  ", "( 10.0, 0.0 )", mono_s)
        self._lbl_target = self._make_data_row(side, "TARGET  ", "( 80.0, 15.0)", mono_s)
        self._lbl_dist   = self._make_data_row(side, "DISTANCE", "—", mono_s)
        sep()

        tk.Label(side, text="[ AI CALCULATION ]", bg=BG, fg=PHOSPHOR2,
                 font=mono, anchor="w").pack(fill="x")
        sep()

        self._lbl_pred   = self._make_data_row(side, "θ PREDICT", "—", mono_s)
        self._lbl_true   = self._make_data_row(side, "θ OPTIMAL", "—", mono_s)
        self._lbl_err    = self._make_data_row(side, "Δθ ERROR ", "—", mono_s)
        sep()

        tk.Label(side, text="[ SHOT RESULT ]", bg=BG, fg=PHOSPHOR2,
                 font=mono, anchor="w").pack(fill="x")
        sep()

        self._lbl_miss   = self._make_data_row(side, "MISS DIST", "—", mono_s)
        self._lbl_result = tk.Label(side, text="READY TO FIRE",
                                    bg=BG, fg=PHOSPHOR, font=mono_l, anchor="w")
        self._lbl_result.pack(fill="x", pady=(4,0))
        sep()

        # FIRE button
        self.fire_btn = tk.Button(
            side, text="[ FIRE ]",
            bg=BG, fg=PHOSPHOR, font=mono_l,
            activebackground=PHOSPHOR, activeforeground=BG,
            relief="flat", bd=0,
            highlightthickness=1, highlightbackground=PHOSPHOR,
            cursor="hand2", pady=10,
            command=self._fire
        )
        self.fire_btn.pack(fill="x", pady=(4,6))

        # hint
        tk.Label(side, text="drag  ◈  to move cannon\ndrag  ✦  to move target",
                 bg=BG, fg=PHOSPHOR3, font=mono_s, justify="left").pack(fill="x")

        # ── bottom bar ──
        tk.Frame(root, bg=PHOSPHOR3, height=1).pack(fill="x", padx=16)
        bot = tk.Frame(root, bg=BG, pady=5)
        bot.pack(fill="x", padx=16)
        self._status = tk.Label(bot, text="SYSTEM READY  ·  DRAG ICONS TO REPOSITION  ·  CLICK FIRE TO LAUNCH",
                                bg=BG, fg=PHOSPHOR3, font=mono_s)
        self._status.pack(side="left")

    def _make_data_row(self, parent, label, value, fnt):
        row = tk.Frame(parent, bg=BG)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label+":", bg=BG, fg=PHOSPHOR3,
                 font=fnt, width=10, anchor="w").pack(side="left")
        lbl = tk.Label(row, text=value, bg=BG, fg=PHOSPHOR, font=fnt, anchor="w")
        lbl.pack(side="left")
        return lbl

    # ── world ↔ canvas coords ───────────────────────────────────────────────

    def _w2c(self, wx, wy):
        """World metres → canvas pixels."""
        cw = self.CANVAS_W - self.PAD_L - self.PAD_R
        ch = self.CANVAS_H - self.PAD_T - self.PAD_B
        cx = self.PAD_L + (wx / self.WORLD_W) * cw
        cy = self.CANVAS_H - self.PAD_B - (wy / self.WORLD_H) * ch
        return cx, cy

    def _c2w(self, cx, cy):
        """Canvas pixels → world metres."""
        cw = self.CANVAS_W - self.PAD_L - self.PAD_R
        ch = self.CANVAS_H - self.PAD_T - self.PAD_B
        wx = (cx - self.PAD_L) / cw * self.WORLD_W
        wy = (self.CANVAS_H - self.PAD_B - cy) / ch * self.WORLD_H
        return wx, wy

    # ── drag ────────────────────────────────────────────────────────────────

    def _on_press(self, ev):
        cx, cy = self._w2c(self.cx0, self.cy0)
        tx, ty = self._w2c(self.tx,  self.ty)
        if math.hypot(ev.x - cx, ev.y - cy) < 18:
            self._drag = 'cannon'
        elif math.hypot(ev.x - tx, ev.y - ty) < 18:
            self._drag = 'target'

    def _on_drag(self, ev):
        if not self._drag:
            return
        wx, wy = self._c2w(ev.x, ev.y)
        wx = max(1, min(self.WORLD_W - 2, wx))
        wy = max(0, min(self.WORLD_H - 2, wy))
        if self._drag == 'cannon':
            self.cx0 = wx
            self.cy0 = 0   # cannon stays on ground
        else:
            self.tx = wx
            self.ty = wy
        self._stop_anim()
        self._draw_scene()

    def _on_release(self, ev):
        self._drag = None

    # ── drawing ─────────────────────────────────────────────────────────────

    def _draw_scene(self, traj_pts=None, traj_progress=1.0):
        c = self.canvas
        c.delete("all")

        # scanlines
        for y in range(0, self.CANVAS_H, 4):
            c.create_line(0, y, self.CANVAS_W, y, fill="#001500", width=1)

        # grid
        cw = self.CANVAS_W - self.PAD_L - self.PAD_R
        ch = self.CANVAS_H - self.PAD_T - self.PAD_B
        steps_x, steps_y = 13, 6
        for i in range(steps_x + 1):
            x = self.PAD_L + i * cw / steps_x
            c.create_line(x, self.PAD_T, x, self.CANVAS_H - self.PAD_B,
                          fill=GRID_COL, dash=(2,4))
        for i in range(steps_y + 1):
            y = self.PAD_T + i * ch / steps_y
            c.create_line(self.PAD_L, y, self.CANVAS_W - self.PAD_R, y,
                          fill=GRID_COL, dash=(2,4))

        # axes
        gx0, gy0 = self._w2c(0, 0)
        gxe, _   = self._w2c(self.WORLD_W, 0)
        _, gye   = self._w2c(0, self.WORLD_H)
        c.create_line(self.PAD_L, gy0, self.CANVAS_W - self.PAD_R, gy0,
                      fill=PHOSPHOR3, width=1)
        c.create_line(self.PAD_L, self.PAD_T, self.PAD_L, self.CANVAS_H - self.PAD_B,
                      fill=PHOSPHOR3, width=1)

        # axis labels
        for m in range(0, self.WORLD_W + 1, 10):
            px, py = self._w2c(m, 0)
            c.create_text(px, py + 14, text=str(m), fill=PHOSPHOR3,
                          font=self._fonts['mono_s'])
        for m in range(0, self.WORLD_H + 1, 10):
            px, py = self._w2c(0, m)
            c.create_text(self.PAD_L - 20, py, text=str(m), fill=PHOSPHOR3,
                          font=self._fonts['mono_s'])

        # axis unit labels
        c.create_text(self.CANVAS_W - self.PAD_R + 10,
                      self.CANVAS_H - self.PAD_B, text="m→", fill=PHOSPHOR3,
                      font=self._fonts['mono_s'])
        c.create_text(self.PAD_L - 20, self.PAD_T - 10, text="m↑", fill=PHOSPHOR3,
                      font=self._fonts['mono_s'])

        # ghost optimal trajectory (faint)
        dx = self.tx - self.cx0
        dy = self.ty - self.cy0
        if dx > 0:
            opt_theta = analytical_theta(dx, dy)
            if opt_theta is not None:
                opt_pts = simulate_trajectory(self.cx0, self.cy0, opt_theta)
                cpx = [self._w2c(p[0], p[1]) for p in opt_pts
                       if 0 <= p[0] <= self.WORLD_W and 0 <= p[1] <= self.WORLD_H]
                if len(cpx) > 1:
                    flat = [v for pt in cpx for v in pt]
                    c.create_line(*flat, fill=DIM, width=1, smooth=True)

        # fired trajectory
        if traj_pts and len(traj_pts) > 1:
            n = max(2, int(len(traj_pts) * traj_progress))
            visible = traj_pts[:n]
            cpx = [self._w2c(p[0], p[1]) for p in visible
                   if -5 <= p[0] <= self.WORLD_W + 5 and -5 <= p[1] <= self.WORLD_H + 5]
            if len(cpx) > 1:
                flat = [v for pt in cpx for v in pt]
                c.create_line(*flat, fill=PHOSPHOR2, width=2, smooth=True)
            # projectile dot at tip
            if cpx:
                ex, ey = cpx[-1]
                c.create_oval(ex-5, ey-5, ex+5, ey+5,
                              fill=PHOSPHOR, outline=PHOSPHOR2, width=1)

        # target crosshair
        tx, ty_c = self._w2c(self.tx, self.ty)
        r = 12
        c.create_line(tx-r, ty_c, tx+r, ty_c, fill=AMBER, width=1, dash=(3,3))
        c.create_line(tx, ty_c-r, tx, ty_c+r, fill=AMBER, width=1, dash=(3,3))
        c.create_oval(tx-r, ty_c-r, tx+r, ty_c+r, outline=AMBER, width=1.5)
        c.create_oval(tx-4, ty_c-4, tx+4, ty_c+4, fill=AMBER, outline="")
        c.create_text(tx + r + 4, ty_c - r, text="TARGET", fill=AMBER,
                      font=self._fonts['mono_s'], anchor="w")

        # cannon icon
        cx_c, cy_c = self._w2c(self.cx0, self.cy0)
        # draw barrel pointing at target if theta known
        barrel_theta = self.fired_theta if self.fired_theta else (
            analytical_theta(self.tx - self.cx0, self.ty - self.cy0) or 0.785
        )
        blen = 22
        bex = cx_c + blen * math.cos(barrel_theta)
        bey = cy_c - blen * math.sin(barrel_theta)
        c.create_line(cx_c, cy_c, bex, bey, fill=PHOSPHOR, width=5,
                      capstyle="round")
        # base
        c.create_rectangle(cx_c - 14, cy_c - 6, cx_c + 14, cy_c + 6,
                           fill=PHOSPHOR3, outline=PHOSPHOR2, width=1)
        c.create_text(cx_c, cy_c + 18, text="◈ CANNON", fill=PHOSPHOR,
                      font=self._fonts['mono_s'])

        # hit flash
        if self.hit and traj_progress >= 1.0:
            hx, hy = self._w2c(self.tx, self.ty)
            for r2, alpha in [(30,"#003300"), (20,"#006600"), (12,"#00aa00")]:
                c.create_oval(hx-r2, hy-r2, hx+r2, hy+r2,
                              fill=alpha, outline=PHOSPHOR, width=1)
            c.create_text(hx, hy - 36, text="⬤ IMPACT", fill=PHOSPHOR,
                          font=self._fonts['mono_l'])

        # update sidebar labels
        self._lbl_cannon.config(
            text=f"( {self.cx0:5.1f}, {self.cy0:4.1f} )")
        self._lbl_target.config(
            text=f"( {self.tx:5.1f}, {self.ty:4.1f} )")
        dist = math.hypot(self.tx - self.cx0, self.ty - self.cy0)
        self._lbl_dist.config(text=f"{dist:6.1f} m")

    # ── fire ────────────────────────────────────────────────────────────────

    def _fire(self):
        if self.anim_id:
            return

        dx = self.tx - self.cx0
        dy = self.ty - self.cy0

        if dx <= 0:
            self._set_status("ERROR: TARGET MUST BE TO THE RIGHT OF CANNON")
            return

        # AI prediction
        with torch.no_grad():
            state = make_state(dx, dy)
            theta_pred = self.model(state).item()

        true_theta = analytical_theta(dx, dy)

        self.fired_theta = theta_pred
        self.true_theta  = true_theta
        self.hit = False

        # sidebar update
        self._lbl_pred.config(text=f"{math.degrees(theta_pred):6.2f}°")
        if true_theta:
            err = abs(math.degrees(theta_pred) - math.degrees(true_theta))
            self._lbl_true.config(text=f"{math.degrees(true_theta):6.2f}°")
            self._lbl_err.config(text=f"{err:6.2f}°",
                                  fg=(PHOSPHOR if err < 2 else AMBER if err < 5 else RED))
        else:
            self._lbl_true.config(text="UNREACHABLE")
            self._lbl_err.config(text="—", fg=PHOSPHOR)

        self._lbl_result.config(text="COMPUTING...", fg=AMBER)
        self._lbl_miss.config(text="—")
        self._set_status(f"AI PREDICTED θ = {math.degrees(theta_pred):.2f}°  ·  LAUNCHING...")
        self.fire_btn.config(state="disabled")

        # simulate trajectory from cannon position
        pts = simulate_trajectory(self.cx0, self.cy0, theta_pred)

        miss = simulate_miss(self.tx, self.ty, theta_pred, x0=self.cx0, y0=self.cy0)
        self.hit = (miss == 0.0)

        self._start_anim(pts, miss)

    def _start_anim(self, pts, miss):
        self.anim_pts  = pts
        self.anim_miss = miss
        self.anim_idx  = 0
        total = len(pts)

        def step():
            if self.anim_idx > total:
                # done
                self._draw_scene(pts, 1.0)
                result_txt  = "⬤ DIRECT HIT!" if self.hit else f"MISS  ({self.anim_miss:.1f}m off)"
                result_col  = PHOSPHOR if self.hit else RED
                miss_txt    = "0.00 m" if self.hit else f"{self.anim_miss:.2f} m"
                self._lbl_result.config(text=result_txt, fg=result_col)
                self._lbl_miss.config(text=miss_txt,
                                       fg=(PHOSPHOR if self.hit else AMBER))
                self._set_status(
                    "DIRECT HIT — AI NAILED IT" if self.hit
                    else f"MISS BY {self.anim_miss:.1f}m — TRY A DIFFERENT POSITION"
                )
                self.fire_btn.config(state="normal")
                self.anim_id = None
                return

            progress = self.anim_idx / max(1, total)
            self._draw_scene(pts, progress)
            self.anim_idx += max(1, total // 80)   # ~80 frames regardless of traj length
            self.anim_id = self.root.after(12, step)

        self.anim_id = self.root.after(10, step)

    def _stop_anim(self):
        if self.anim_id:
            self.root.after_cancel(self.anim_id)
            self.anim_id = None
        self.fired_theta = None
        self.hit = False
        self.fire_btn.config(state="normal")
        self._lbl_result.config(text="READY TO FIRE", fg=PHOSPHOR)
        self._lbl_pred.config(text="—")
        self._lbl_true.config(text="—")
        self._lbl_err.config(text="—", fg=PHOSPHOR)
        self._lbl_miss.config(text="—")

    def _set_status(self, msg):
        self._status.config(text=msg)


# ── Training progress window ────────────────────────────────────────────────

class TrainingWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AimNet — Training")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        try:
            mono   = tkfont.Font(family="Courier New", size=10, weight="bold")
            mono_s = tkfont.Font(family="Courier New", size=8)
            mono_xl= tkfont.Font(family="Courier New", size=14, weight="bold")
        except Exception:
            mono   = tkfont.Font(size=10, weight="bold")
            mono_s = tkfont.Font(size=8)
            mono_xl= tkfont.Font(size=14, weight="bold")

        self._fonts = dict(mono=mono, mono_s=mono_s, mono_xl=mono_xl)

        tk.Label(self.root, text="AIMNET — SUPERVISED TRAINING",
                 bg=BG, fg=PHOSPHOR, font=mono_xl, pady=12).pack()
        tk.Frame(self.root, bg=PHOSPHOR3, height=1).pack(fill="x", padx=20)

        # log area
        frame = tk.Frame(self.root, bg=BG, padx=20, pady=10)
        frame.pack(fill="both", expand=True)

        self.log = tk.Text(frame, bg=BG2, fg=PHOSPHOR, font=mono_s,
                           width=58, height=18, relief="flat",
                           highlightthickness=1, highlightbackground=PHOSPHOR3,
                           state="disabled", cursor="arrow")
        self.log.pack(fill="both", expand=True)

        # progress bar
        pb_frame = tk.Frame(self.root, bg=BG, padx=20, pady=8)
        pb_frame.pack(fill="x")
        tk.Label(pb_frame, text="PROGRESS:", bg=BG, fg=PHOSPHOR3,
                 font=mono_s).pack(side="left")
        self.pb_canvas = tk.Canvas(pb_frame, bg=BG2, height=14, width=380,
                                    highlightthickness=1,
                                    highlightbackground=PHOSPHOR3)
        self.pb_canvas.pack(side="left", padx=(8,0))
        self._pct_lbl = tk.Label(pb_frame, text="  0%", bg=BG, fg=PHOSPHOR,
                                  font=mono_s, width=5)
        self._pct_lbl.pack(side="left")

        self._append("INITIALISING AIMNET...\n")
        self._append("architecture: Linear(3→256) × 3 → Linear(256→1)\n")
        self._append("loss:         MSE\n")
        self._append("optimiser:    Adam  lr=3e-4\n")
        self._append("scheduler:    CosineAnnealingLR\n")
        self._append("─" * 50 + "\n")

    def _append(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    def on_progress(self, step, total, loss, err_deg):
        bar = "█" * int(step/total*40) + "░" * (40 - int(step/total*40))
        pct = int(step/total*100)
        self.root.after(0, lambda: self._append(
            f"step {step:>5}/{total}  loss={loss:.8f}  err={err_deg:.3f}°\n"))
        self.root.after(0, lambda: self._update_bar(pct))

    def _update_bar(self, pct):
        self.pb_canvas.delete("all")
        w = int(380 * pct / 100)
        self.pb_canvas.create_rectangle(0, 0, w, 14, fill=PHOSPHOR2, outline="")
        self._pct_lbl.config(text=f"{pct:3d}%")

    def on_done(self, model):
        self.root.after(0, lambda: self._finish(model))

    def _finish(self, model):
        self._append("─" * 50 + "\n")
        self._append("TRAINING COMPLETE\n")
        self._append("opening targeting terminal...\n")
        self._update_bar(100)
        self.root.after(1200, lambda: self._launch_terminal(model))

    def _launch_terminal(self, model):
        self.root.destroy()
        term = tk.Tk()
        app  = BallisticTerminal(term, model)
        term.mainloop()

    def run_training(self):
        def worker():
            train(
                steps=8000,
                on_progress=self.on_progress,
                on_done=self.on_done
            )
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        self.root.mainloop()


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tw = TrainingWindow()
    tw.run_training()