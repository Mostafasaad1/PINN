
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║          THE BALL-DROP ADVENTURE: A Physics-Informed Neural Network         ║
# ║                    Explained for 7-year-olds (and curious adults)           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# ── THE STORY ──────────────────────────────────────────────────────────────────
#
#  Meet NETTIE, a little kid who wants to be the world's best ball-guesser.
#  You throw a ball straight UP into the air. It flies up... then curves back down.
#  Nettie's job: at ANY moment in time, GUESS where that ball is.
#
#  Nettie has TWO teachers:
#
#   📺  THE VIDEO TEACHER  (→ "Data Loss")
#       Nettie watches a video of the actual ball flying.
#       The video only has TEN frames (like only 10 blurry photos).
#       If Nettie's guess doesn't match the photos → she gets scolded.
#       This scolding is called the DATA LOSS.
#
#   👨  DAD  (→ "Physics Loss")
#       Dad knows the UNBREAKABLE RULES OF THE PLAYGROUND.
#       His most important rule is: "Gravity ALWAYS pulls the ball down.
#       Every single second, the ball's downward speed increases by 9.8 m/s.
#       No exceptions. Not even on Tuesdays."
#       In math, Dad's rule looks like: d²y/dt² = −g
#       (Which just means: the "change-in-the-change of height" = −9.8)
#       If Nettie's guesses break this rule ANYWHERE → Dad scolds her more.
#       This scolding is called the PHYSICS LOSS.
#
#   🏆  A PINN (Physics-Informed Neural Network)
#       = Nettie learning from BOTH the video AND Dad's rules at the same time.
#       Result: she only needs a FEW video frames, but gets the WHOLE flight right!
#
# ── THE PHYSICS (Dad's Rule, Simply) ──────────────────────────────────────────
#
#   Real answer: y(t) = y₀ + v₀·t − ½·g·t²
#   - y₀ = 10 m  (start height, like throwing from a balcony)
#   - v₀ = 5 m/s (thrown UPWARD)
#   - g  = 9.8 m/s²  (gravity, always pulling DOWN)
#   Dad's rule (the "ODE"): d²y/dt² + g = 0
#   → The ball's acceleration + gravity must always equal ZERO.
#
# ── HOW TO RUN ─────────────────────────────────────────────────────────────────
#   pip install torch numpy matplotlib
#   python pinn_ball_drop.py
#
# ══════════════════════════════════════════════════════════════════════════════

import torch          # The main "brain-building" library (like LEGO for AI)
import torch.nn as nn # nn = "neural network" toolbox inside PyTorch
import numpy as np    # For fast math with arrays of numbers
import math           # For sqrt (square root) to find flight time
import matplotlib.pyplot as plt  # For drawing a pretty picture at the end

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: SET UP THE PLAYGROUND
# ══════════════════════════════════════════════════════════════════════════════

# "Dad's Rule #1: How hard gravity pulls everything down."
# On Earth this is 9.8 metres per second, every second.
g = 9.8   # m/s²

# "Where does the story begin?"
y0 = 10.0  # Ball starts 10 metres high (like standing on a 3-story balcony)
v0 = 5.0   # Ball is thrown UPWARD at 5 m/s

# "How long is the ball in the air before it hits the ground?"
# We solve y(t) = 0  →  0 = y0 + v0·t − ½·g·t²
# Using the quadratic formula: t = (v0 + sqrt(v0² + 2·g·y0)) / g
t_end = (v0 + math.sqrt(v0**2 + 2.0 * g * y0)) / g  # ≈ 2.03 seconds

# "Which computer chip should Nettie's brain live on?"
# 'cpu' = normal computer chip  |  'cuda' = fancy gaming GPU (much faster)
device = torch.device('cpu')

print("=" * 65)
print("  ⚽  THE BALL-DROP ADVENTURE: Physics-Informed Neural Network")
print("=" * 65)
print(f"\n🕐  The ball is in the air for {t_end:.3f} seconds.")
print(f"🏠  Start height: {y0} m  |  Throw speed: {v0} m/s  |  Gravity: {g} m/s²\n")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: THE REAL ANSWER (The Ground Truth)
# ══════════════════════════════════════════════════════════════════════════════

def true_solution(t):
    """
    This is the PERFECT video — what physics says the ball REALLY does.
    Given any time t, it returns the exact height.

    Formula: y(t) = y0 + v0·t − ½·g·t²
    Kid version: "Height = Start + (Throw-Speed × Time) − (Half × Gravity × Time²)"

    We use this to:
      1) Make our fake blurry "training video" (a few snapshots)
      2) Check how accurate Nettie's final guesses are
    """
    return y0 + v0 * t - 0.5 * g * t ** 2

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: MAKE THE TRAINING DATA  (The Blurry Video Snapshots)
# ══════════════════════════════════════════════════════════════════════════════

# "How many video snapshots do we show Nettie?"
# We only give her 10 — just 10 paused frames from the whole flight!
# If we gave ALL the data, she wouldn't need to learn Dad's rules.
num_snapshots = 10

# torch.linspace(start, end, n) → makes n evenly-spaced numbers from start to end
# Like putting n tick marks on a ruler between two values.
t_data = torch.linspace(0, t_end, num_snapshots, device=device).unsqueeze(1)
# .unsqueeze(1) turns  [t1, t2, ..., t10]  into  [[t1],[t2],...,[t10]]
# (Each time is on its own row — like a neat column in a table)

# "What height does the true physics say the ball is at in each snapshot?"
y_data_clean = true_solution(t_data)

# "Add a tiny bit of blur/noise — real videos are never perfectly sharp."
# torch.randn_like makes random fuzz the same shape as our data.
# Multiply by 0.1 so the fuzz is small (only ±10 cm, roughly).
torch.manual_seed(42)  # "Use this exact random seed so results are repeatable"
noise = 0.1 * torch.randn_like(y_data_clean)
y_data = y_data_clean + noise  # Blurry snapshot heights

print(f"📸  Created {num_snapshots} blurry video snapshots for Nettie.")
print(f"    First frame:  t = {t_data[0].item():.3f} s  →  height ≈ {y_data[0].item():.2f} m")
print(f"    Last frame:   t = {t_data[-1].item():.3f} s  →  height ≈ {y_data[-1].item():.2f} m")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MAKE THE PHYSICS CHECK POINTS  (Where Dad Inspects the Rules)
# ══════════════════════════════════════════════════════════════════════════════

# "Dad doesn't just check the 10 snapshots — he checks EVERYWHERE."
# We'll create 50 evenly-spaced moments for Dad to inspect.
# These are called "collocation points" in fancy math, but let's call them
# "Dad's inspection spots."
num_physics_pts = 100

# requires_grad=True is MAGIC SAUCE.
# It tells PyTorch: "Put a speedometer on every one of these time values
# so we can later ask 'how does Nettie's height guess CHANGE as time changes?'"
# Without this, we can't compute the velocity or acceleration of Nettie's guesses.
t_physics = (
    torch.linspace(0, t_end, num_physics_pts, device=device)
    .unsqueeze(1)          # Make it a column, same as t_data
    .requires_grad_(True)  # ← THE MAGIC SPEEDOMETER
)

print(f"\n🔍  Dad will check his rules at {num_physics_pts} inspection spots.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BUILD NETTIE'S BRAIN  (The Neural Network)
# ══════════════════════════════════════════════════════════════════════════════
#
# Nettie's brain takes ONE number in (a time t)
# and outputs ONE number (her height guess y).
#
# In between, she has "hidden layers" — think of these as scratch-paper steps.
# Each layer takes the previous ideas and mixes them into new, deeper ideas.
#
# Architecture:
#   Input (1) → Layer1 (32 neurons) → Layer2 (32) → Layer3 (32) → Output (1)
#
# "Neuron" = one tiny calculation unit, like one brain cell doing simple math.
# 32 neurons per layer is enough for a smooth curved function like a ball flight.

class NettieTheBallGuesser(nn.Module):
    """
    Nettie's brain blueprint.

    nn.Module is the base class every PyTorch brain must inherit from.
    Think of it as "the factory template for all brains."
    """

    def __init__(self):
        """
        Called when Nettie is first 'born.'
        We wire up all her brain layers here.
        """
        # "Set up the basic brain stuff that PyTorch always needs first."
        super(NettieTheBallGuesser, self).__init__()

        # nn.Sequential is like a factory conveyor belt:
        # plug in a time → it flows through every station in order → comes out a height.
        self.brain = nn.Sequential(

            # ── Station 1 ──────────────────────────────────────────────────
            # nn.Linear(1, 32) = "Take 1 number in, produce 32 different thoughts."
            # Each of the 32 neurons multiplies your input by its own weight,
            # adds its own bias, and passes it forward. Like 32 kids each
            # solving the same problem a slightly different way.
            nn.Linear(1, 32),

            # nn.Tanh() = the "brain squisher."
            # Without it, numbers could grow to infinity and the brain goes crazy.
            # Tanh squishes ANYTHING into a range between −1 and +1.
            # This lets the brain learn curved, wiggly shapes (like a ball path!).
            nn.Tanh(),

            # ── Station 2 ──────────────────────────────────────────────────
            # "Take those 32 thoughts and think even deeper — make 32 new thoughts."
            nn.Linear(32, 32),
            nn.Tanh(),

            # ── Station 3 ──────────────────────────────────────────────────
            # "Think even deeper!"
            nn.Linear(32, 32),
            nn.Tanh(),

            # ── Output Station ─────────────────────────────────────────────
            # "Okay Nettie, after all that thinking, give ONE final height guess."
            # No Tanh here — the height can be any number (above or below ground).
            nn.Linear(32, 1),
        )

    def forward(self, t):
        """
        Called every time Nettie makes a guess.
        You hand her a time t → she runs it through her brain → returns a height.

        'forward' is called 'forward' because info moves FORWARD through layers
        (left to right on the conveyor belt), not backward.
        """
        # "Run the time through all of Nettie's stations. Get a height back."
        return self.brain(t)


# "Wake Nettie up!  Create one instance of her brain."
nettie = NettieTheBallGuesser().to(device)
# .to(device) = "put Nettie's brain on the computer chip we chose earlier"

# Count how many "knobs" (parameters = weights and biases) Nettie has.
# Each knob gets tweaked a little during training to improve her guesses.
total_params = sum(p.numel() for p in nettie.parameters())

print(f"\n🧠  Nettie is awake!  She has {total_params} brain-knobs to tune.")
print("    Network layout:", nettie)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SET UP THE TEACHING TOOLS
# ══════════════════════════════════════════════════════════════════════════════

# ── The Grading Ruler (Loss Function) ───────────────────────────────────────
#
# nn.MSELoss() = "Mean Squared Error" loss.
# "Squared Error" means: if your guess is 3 metres off, you get 9 penalty points
# (3² = 9). If you're 10 metres off, you get 100 penalty points (10² = 100).
# Big mistakes are punished WAY more than small ones — so Nettie really wants
# to avoid huge errors!
# "Mean" means we average all those penalties into one single score.
loss_fn = nn.MSELoss()

# ── The Coach (Optimizer) ────────────────────────────────────────────────────
#
# After every practice round, the coach looks at Nettie's mistakes and
# adjusts ALL her brain-knobs to make her a little better.
#
# torch.optim.Adam is a smart coach that:
#   - Adjusts each knob by a different amount depending on how useful it is
#   - Uses a "memory" of past adjustments to avoid zig-zagging
#
# lr = "learning rate" = how BIG each adjustment is.
# lr=0.001 means each knob moves by at most 0.1% in one step.
# Too big → Nettie overshoots and never settles.
# Too small → Nettie takes forever to learn.
# 0.001 is the "Goldilocks" rate for most problems.
optimizer = torch.optim.Adam(nettie.parameters(), lr=0.001)
# nettie.parameters() = hand the coach the list of all Nettie's knobs

# ── Training Settings ────────────────────────────────────────────────────────
num_epochs = 8000  # "How many total practice rounds does Nettie get?"

# λ (lambda) values: how much weight to give each teacher's scolding.
# If lambda_data = 1.0 → Video Teacher's feedback counts fully.
# If lambda_physics = 1.0 → Dad's rule feedback also counts fully.
# They're equal here — both teachers matter equally!
lambda_data    = 1.0
lambda_physics = 1.0

# ── Logging ──────────────────────────────────────────────────────────────────
# We'll record losses over time so we can plot them at the end.
history = {"total": [], "data": [], "physics": []}

print(f"\n🏋️   Training plan:  {num_epochs} epochs,  "
      f"λ_data={lambda_data},  λ_physics={lambda_physics}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: THE TRAINING LOOP  (The Practice Sessions!)
# ══════════════════════════════════════════════════════════════════════════════
#
# In each "epoch" (one full practice round), Nettie:
#   A. Wipes the old scolding marks
#   B. Looks at the 10 video snapshots → gets a DATA scolding
#   C. Dad checks her guesses at 50 spots → gets a PHYSICS scolding
#   D. We add the two scoldings together → TOTAL scolding
#   E. The total scolding travels BACKWARD through her brain (backprop)
#      to figure out which knobs caused the mistakes
#   F. The coach nudges those knobs to reduce the scolding next time
#
# Repeat thousands of times → Nettie gradually gets better and better!

print("\n" + "=" * 65)
print("  🎮  TRAINING BEGINS!  Watch Nettie learn...")
print("=" * 65 + "\n")

for epoch in range(1, num_epochs + 1):

    # ── A. WIPE THE WHITEBOARD ───────────────────────────────────────────────
    # Before each new practice round, erase the old scolding scores.
    # If we skip this, old scores pile up and give the coach wrong directions.
    # Think of it as clearing your scratchpad before a new math problem.
    optimizer.zero_grad()

    # ══════════════════════════════════════════════════════════════════════
    # B. DATA LOSS — The Video Teacher's Scolding
    # ══════════════════════════════════════════════════════════════════════

    # "Nettie, where do you think the ball is at each of the 10 snapshot times?"
    # We pass all 10 times through Nettie's brain at once and get 10 height guesses.
    y_pred_data = nettie(t_data)
    # y_pred_data  shape: [10, 1]  (one height guess per snapshot time)

    # "How wrong was Nettie compared to the real (blurry) video frames?"
    # loss_data is one positive number: 0 = perfect, huge = very wrong.
    loss_data = loss_fn(y_pred_data, y_data)

    # ══════════════════════════════════════════════════════════════════════
    # C. PHYSICS LOSS — Dad's Rule Checking
    # ══════════════════════════════════════════════════════════════════════
    #
    # Dad's rule:  d²y/dt² = −g
    #
    # What does d²y/dt² mean in kid language?
    #   • "d" means "a tiny little change in..."
    #   • "y" is height,  "t" is time
    #   • dy/dt  = "how much does the HEIGHT change when time ticks forward a hair?"
    #              → That's VELOCITY (speed up or down)
    #   • d²y/dt² = "how much does the VELOCITY change when time ticks forward a hair?"
    #              → That's ACCELERATION
    # Dad says: acceleration must ALWAYS equal exactly −9.8 m/s².
    #
    # PyTorch's torch.autograd.grad is the MAGIC CALCULATOR that
    # uses those "speedometers" (requires_grad=True) we installed on t_physics.
    # It figures out these rates of change automatically — no pencil and paper needed!

    # "Nettie, guess the height at all 50 of Dad's inspection spots."
    y_pred_phys = nettie(t_physics)
    # y_pred_phys shape: [50, 1]  (one height guess per inspection spot)

    # ── Compute dy/dt  (VELOCITY = "how fast is height changing?") ──────────
    #
    # torch.autograd.grad arguments:
    #   outputs    = what we want the derivative OF        → height guesses
    #   inputs     = what we differentiate WITH RESPECT TO → time
    #   grad_outputs = torch.ones_like(...)  → "compute gradient for EVERY row"
    #                  (otherwise it only works for a single scalar output)
    #   create_graph = True  → "keep the math around so we can differentiate AGAIN"
    #                          (we need this to get the second derivative)
    #   retain_graph = True  → "don't delete the computation graph yet"
    dy_dt = torch.autograd.grad(
        outputs      = y_pred_phys,
        inputs       = t_physics,
        grad_outputs = torch.ones_like(y_pred_phys),
        create_graph = True,
        retain_graph = True,
    )[0]
    # [0] grabs just the gradient tensor from the returned tuple.
    # dy_dt shape: [50, 1]  ← the velocity at each of Dad's 50 spots

    # ── Compute d²y/dt²  (ACCELERATION = "how fast is velocity changing?") ──
    #
    # Same trick again, but now we differentiate dy_dt instead of y.
    # The result should equal −9.8 everywhere if Nettie is obeying the rules.
    d2y_dt2 = torch.autograd.grad(
        outputs      = dy_dt,
        inputs       = t_physics,
        grad_outputs = torch.ones_like(dy_dt),
        create_graph = True,
        retain_graph = True,
    )[0]
    # d2y_dt2 shape: [50, 1]  ← the acceleration at each of Dad's 50 spots

    # ── Check Dad's Rule ────────────────────────────────────────────────────
    #
    # Dad's rule says: d²y/dt² + g = 0
    # Rearranged: d²y/dt² = −g  (acceleration = negative gravity)
    #
    # "residual" = how much Nettie's guesses BREAK the rule.
    # Perfect obedience → residual is 0 everywhere.
    # Breaking the rule → residual is non-zero → Dad scolds!
    physics_residual = d2y_dt2 + g
    # ↑ If Nettie perfectly follows gravity, every entry here is 0.0
    #   If she guesses crazy heights, these entries will be big numbers.

    # "How much did Nettie break Dad's rule, on average (squared)?"
    # We compare the residual to a tensor of all-zeros (the perfect answer).
    loss_physics = loss_fn(physics_residual, torch.zeros_like(physics_residual))

    # ══════════════════════════════════════════════════════════════════════
    # D. TOTAL LOSS — Both Scoldings Added Up
    # ══════════════════════════════════════════════════════════════════════

    # "Nettie's total report card = (how wrong vs video) + (how much she broke physics)"
    # Multiplying by lambda lets us decide how much each teacher's score counts.
    total_loss = lambda_data * loss_data + lambda_physics * loss_physics

    # Save to history (for plotting later)
    history["total"].append(total_loss.item())
    history["data"].append(loss_data.item())
    history["physics"].append(loss_physics.item())
    # .item() turns a PyTorch tensor with one number into a plain Python float

    # ══════════════════════════════════════════════════════════════════════
    # E. BACKPROPAGATION — Tracing Mistakes Backward Through the Brain
    # ══════════════════════════════════════════════════════════════════════
    #
    # total_loss.backward() is where the REAL learning happens.
    #
    # It asks: "Which brain-knob caused how much of this mistake?"
    # It traces the error BACKWARD from the final loss, through every layer,
    # all the way back to each individual knob.
    #
    # Kid analogy: imagine you spilled juice on the floor.
    # Instead of just mopping, you trace back WHICH glass you bumped → fix that.
    # .backward() does exactly this, automatically, for all knobs at once.
    total_loss.backward()

    # ══════════════════════════════════════════════════════════════════════
    # F. OPTIMIZER STEP — The Coach Nudges the Knobs
    # ══════════════════════════════════════════════════════════════════════
    #
    # optimizer.step() reads all the gradient info computed by .backward()
    # and adjusts every knob by a small amount in the direction that
    # REDUCES the loss.
    #
    # Knob caused big mistake? → Big nudge.
    # Knob barely mattered? → Tiny nudge.
    # Learning rate (0.001) is the maximum nudge size.
    optimizer.step()

    # ══════════════════════════════════════════════════════════════════════
    # G. THE STORY — Print Progress Updates
    # ══════════════════════════════════════════════════════════════════════

    if epoch % 1000 == 0 or epoch == 1:
        tl = total_loss.item()
        dl = loss_data.item()
        pl = loss_physics.item()

        # Pick a story beat based on how well Nettie is doing so far
        if epoch == 1:
            story  = "Nettie just woke up. Her guesses are RANDOM nonsense. 😴"
            feeling = "She doesn't even know what a ball is yet!"
        elif tl > 5.0:
            story  = "Nettie is guessing wildly, all over the place! 🙈"
            feeling = "She sees the video but it's making no sense yet."
        elif tl > 1.0:
            story  = "Something is clicking... Nettie sees the shape! 🤔"
            feeling = "The video is starting to look familiar. Dad is watching."
        elif tl > 0.1:
            story  = "Nettie is using BOTH the video AND Dad's gravity rules now! 📚"
            feeling = "She mumbles: 'It goes up... then gravity pulls it down...'"
        elif tl > 0.01:
            story  = "Nettie's guesses are really good! Dad smiles a little. 😊"
            feeling = "Physics and data are working together beautifully!"
        elif tl > 0.001:
            story  = "Nettie is NAILING it. The ball path is almost perfect! 🎯"
            feeling = "She can now predict where the ball goes between the snapshots!"
        else:
            story  = "Nettie has MASTERED the ball. BOTH teachers are SO proud! 🏆⭐"
            feeling = "She follows the video AND gravity rules at the same time!"

        print(f"📅  Epoch {epoch:5d} / {num_epochs}")
        print(f"   {story}")
        print(f"   💭 {feeling}")
        print(f"   📺 Video scolding  (data loss):    {dl:.6f}")
        print(f"   👨 Dad's scolding  (physics loss): {pl:.6f}")
        print(f"   📊 Total scolding  (total loss):   {tl:.6f}")
        print()

# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  🎉  TRAINING COMPLETE!  Nettie has graduated!")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: THE FINAL EXAM  (How Well Did Nettie Learn?)
# ══════════════════════════════════════════════════════════════════════════════

print("\n📝  FINAL EXAM — Testing Nettie on 100 moments (she only trained on 10!)\n")

# Switch Nettie to "evaluation mode."
# Some layers behave slightly differently while training vs testing.
# eval() says: "Stop training. Just use what you know."
nettie.eval()

# torch.no_grad() tells PyTorch: "We're just testing, no need for speedometers."
# This makes inference faster because PyTorch skips tracking gradients.
with torch.no_grad():

    # Create 100 evenly-spaced test times (10× more than Nettie trained on!)
    t_test = torch.linspace(0, t_end, 100, device=device).unsqueeze(1)

    # "Nettie, where is the ball at each of these 100 moments?"
    y_pred_test = nettie(t_test)

    # What the REAL physics says the heights should be
    y_true_test = true_solution(t_test)

    # "How far off (in metres) is Nettie's guess from the real height?"
    # torch.abs() = absolute value (so we don't mix up positive and negative errors)
    abs_errors  = torch.abs(y_pred_test - y_true_test)
    avg_error   = torch.mean(abs_errors).item()
    max_error   = torch.max(abs_errors).item()

print(f"  📊 Average error:  {avg_error:.4f} metres")
print(f"  📊 Max error:      {max_error:.4f} metres")

if avg_error < 0.05:
    grade = "🏆  PERFECT!  Nettie is within 5 cm on average. She's a genius!"
elif avg_error < 0.2:
    grade = "✅  GREAT!  Nettie is within 20 cm. The physics really helped!"
elif avg_error < 0.5:
    grade = "👍  GOOD.   Nettie learned the basics. More epochs would help."
else:
    grade = "📚  HMM.   Try increasing num_epochs to 10000+ and re-run."

print(f"\n  {grade}")

# ── Comparison Table ─────────────────────────────────────────────────────────
print("\n" + "─" * 58)
print(f"{'Time (s)':>9}  {'Real Height (m)':>16}  {'Nettie Guess (m)':>17}  {'Error (m)':>9}")
print("─" * 58)

with torch.no_grad():
    # Show a table of 11 moments (start, middle, end and a few in between)
    t_show = torch.linspace(0, t_end, 11, device=device).unsqueeze(1)
    y_show_pred = nettie(t_show).numpy()
    y_show_true = true_solution(t_show).numpy()

    for i in range(11):
        t_val  = t_show[i].item()
        y_true = y_show_true[i, 0]
        y_pred = y_show_pred[i, 0]
        err    = abs(y_true - y_pred)
        flag   = " ← training point nearby" if i % 1 == 0 else ""
        print(f"{t_val:>9.3f}  {y_true:>16.4f}  {y_pred:>17.4f}  {err:>9.4f}")

print("─" * 58)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: PRETTY PLOTS  (So we can SEE what happened)
# ══════════════════════════════════════════════════════════════════════════════

print("\n📈  Drawing the results...")

# Create a 1×2 grid of plots (left = ball path, right = loss history)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Nettie the PINN Learns to Track a Flying Ball", fontsize=14, fontweight="bold")

# ── Left Plot: Ball Path ──────────────────────────────────────────────────────

# Dense time grid for smooth curves
t_plot = np.linspace(0, t_end, 300)
t_plot_tensor = torch.tensor(t_plot, dtype=torch.float32, device=device).unsqueeze(1)

nettie.eval()
with torch.no_grad():
    y_nettie = nettie(t_plot_tensor).numpy().flatten()

y_real = y0 + v0 * t_plot - 0.5 * g * t_plot ** 2

# True physics path (solid blue line)
ax1.plot(t_plot, y_real, "b-", linewidth=2.5, label="📐 Real Physics Answer")

# Nettie's learned path (dashed red line)
ax1.plot(t_plot, y_nettie, "r--", linewidth=2.5, label="🤖 Nettie's Guess (PINN)")

# The 10 blurry training snapshots (black crosses)
ax1.scatter(
    t_data.numpy().flatten(),
    y_data.numpy().flatten(),
    color="black", zorder=5, s=80, marker="x", linewidths=2.5,
    label=f"📸 Video Snapshots ({num_snapshots} blurry frames)"
)

# Shade below the x-axis as "ground"
ax1.axhline(0, color="saddlebrown", linewidth=1.5, linestyle="-")
ax1.fill_between(t_plot, 0, -0.5, color="saddlebrown", alpha=0.2, label="🌍 Ground")

ax1.set_xlabel("Time (seconds)", fontsize=11)
ax1.set_ylabel("Height (metres)", fontsize=11)
ax1.set_title("Ball Flight:  Real vs Nettie's PINN Guess", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.5, max(y_real) * 1.25)

# ── Right Plot: Loss History ──────────────────────────────────────────────────

epochs_x = list(range(1, num_epochs + 1))

ax2.semilogy(epochs_x, history["total"],   "k-",  linewidth=1.8, label="📊 Total Loss")
ax2.semilogy(epochs_x, history["data"],    "b--", linewidth=1.5, label="📺 Data Loss (Video Teacher)")
ax2.semilogy(epochs_x, history["physics"], "r:",  linewidth=1.5, label="👨 Physics Loss (Dad's Rules)")

ax2.set_xlabel("Epoch (Practice Round)", fontsize=11)
ax2.set_ylabel("Loss (log scale — smaller = better!)", fontsize=11)
ax2.set_title("How Nettie's Scolding Decreases Over Time", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, which="both")

plt.tight_layout()
plt.savefig("pinn_results.png", dpi=140, bbox_inches="tight")
print("💾  Saved plot to:  pinn_results.png")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: THE MORAL OF THE STORY
# ══════════════════════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════════════╗
║                  🎓  THE MORAL OF THE STORY  🎓                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Nettie learned to guess the ball's position perfectly using:    ║
║                                                                  ║
║  📺  THE VIDEO  (just 10 blurry snapshots!)  →  Data Loss        ║
║      "I saw the ball HERE at these exact moments."               ║
║                                                                  ║
║  👨  DAD'S RULES  (gravity, everywhere!) →  Physics Loss         ║
║      "d²y/dt² + g = 0  means:                                    ║
║       Nettie's acceleration MUST equal −9.8 m/s² always."        ║
║                                                                  ║
║  ─────────────────────────────────────────────────────────────   ║
║  ❌  Without data only:  Nettie might fit ANY smooth curve!       ║
║  ❌  Without physics only:  Nettie needs THOUSANDS of snapshots! ║
║  ✅  WITH BOTH (= PINN!):  10 snapshots + gravity → PERFECT!     ║
║                                                                  ║
║  This is the SUPERPOWER of Physics-Informed Neural Networks!     ║
║  They let AI learn physical systems with tiny amounts of data,   ║
║  because the laws of physics fill in all the gaps.               ║
║                                                                  ║
║  Real PINNs are used in:                                         ║
║   🚀 Rocket trajectory prediction                                ║
║   🌊 Fluid flow simulation (weather, blood flow)                 ║
║   🔥 Heat transfer in jet engines                                ║
║   🏥 Modelling how medicine spreads in the body                  ║
╚══════════════════════════════════════════════════════════════════╝
""")
