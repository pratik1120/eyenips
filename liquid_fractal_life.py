import taichi as ti
import math
import numpy as np
from dataclasses import dataclass
import threading
import json
import tkinter as tk
from tkinter import ttk
import time

# Initialize Taichi
ti.init(arch=ti.gpu)

# ============================================
# CUSTOMIZABLE PARAMETERS - Change these!
# ============================================

# Screen resolution
WIDTH, HEIGHT = 1200, 800

# Particle count
NUM_PARTICLES = 500000

# Color palette
COLOR_MODE = "rainbow"  # "rainbow", "plasma", "ocean", "fire", "custom"

# ============== EQUATIONS - Modify these to change behavior ==============
# These equations control how particles move through space

# Equation 1: Primary flow field influence (x-component)
def eq1(x, y, t):
    return math.sin(x * 0.01 + t * 0.5) * math.cos(y * 0.005) + math.sin(t * 0.3) * 0.5

# Equation 2: Secondary flow field influence (y-component)
def eq2(x, y, t):
    return math.cos(x * 0.008 - t * 0.4) * math.sin(y * 0.01) + math.cos(t * 0.25) * 0.5

# Equation 3: Fractal dimension (affects swirling/spiraling)
def eq3(x, y, t):
    dist = math.sqrt((x - WIDTH/2)**2 + (y - HEIGHT/2)**2)
    return math.sin(dist * 0.001 - t * 0.2) * 0.8 + math.cos(t * 0.15)

# Equation 4: Radial force (affects clustering)
def eq4(x, y, t):
    return math.atan2(y - HEIGHT/2, x - WIDTH/2) * 0.1 + math.sin(t * 0.1)

# ========================================================================

# Taichi fields
particles_x = ti.field(ti.f32, NUM_PARTICLES)
particles_y = ti.field(ti.f32, NUM_PARTICLES)
particles_vx = ti.field(ti.f32, NUM_PARTICLES)
particles_vy = ti.field(ti.f32, NUM_PARTICLES)
particles_age = ti.field(ti.f32, NUM_PARTICLES)
particles_color = ti.field(ti.f32, NUM_PARTICLES)

# Canvas for rendering
canvas = ti.field(ti.f32, (WIDTH, HEIGHT))

# Parameter fields (single-value fields so kernels can read them)
p_fx_x = ti.field(ti.f32, ())
p_fx_y = ti.field(ti.f32, ())
p_fx_t = ti.field(ti.f32, ())
p_fx_amp = ti.field(ti.f32, ())

p_fy_x = ti.field(ti.f32, ())
p_fy_y = ti.field(ti.f32, ())
p_fy_t = ti.field(ti.f32, ())
p_fy_amp = ti.field(ti.f32, ())

p_spiral_scale = ti.field(ti.f32, ())
p_spiral_t = ti.field(ti.f32, ())
p_radial_strength = ti.field(ti.f32, ())
p_global_speed = ti.field(ti.f32, ())
p_preset = ti.field(ti.i32, ())

# Default parameter values (used to initialize the ti fields and GUI)
DEFAULT_PARAMS = {
    "p_fx_x": 0.01,
    "p_fx_y": 0.005,
    "p_fx_t": 0.5,
    "p_fx_amp": 0.5,
    "p_fy_x": 0.008,
    "p_fy_y": 0.01,
    "p_fy_t": 0.4,
    "p_fy_amp": 0.5,
    "p_spiral_scale": 0.001,
    "p_spiral_t": 0.2,
    "p_radial_strength": 0.1,
    "p_global_speed": 1.0,
    "preset": 0,
}


def init_params():
    # write defaults into Taichi scalar fields
    p_fx_x[None] = DEFAULT_PARAMS["p_fx_x"]
    p_fx_y[None] = DEFAULT_PARAMS["p_fx_y"]
    p_fx_t[None] = DEFAULT_PARAMS["p_fx_t"]
    p_fx_amp[None] = DEFAULT_PARAMS["p_fx_amp"]

    p_fy_x[None] = DEFAULT_PARAMS["p_fy_x"]
    p_fy_y[None] = DEFAULT_PARAMS["p_fy_y"]
    p_fy_t[None] = DEFAULT_PARAMS["p_fy_t"]
    p_fy_amp[None] = DEFAULT_PARAMS["p_fy_amp"]

    p_spiral_scale[None] = DEFAULT_PARAMS["p_spiral_scale"]
    p_spiral_t[None] = DEFAULT_PARAMS["p_spiral_t"]
    p_radial_strength[None] = DEFAULT_PARAMS["p_radial_strength"]
    p_global_speed[None] = DEFAULT_PARAMS["p_global_speed"]
    p_preset[None] = DEFAULT_PARAMS["preset"]




# Kernel-callable versions of the editable equations.
# Edit these `@ti.func` functions to change particle behavior on the GPU.
@ti.func
def eq1_ti(x, y, t):
    # selectable presets for very different behaviors
    val = 0.0
    if p_preset[None] == 0:
        val = ti.sin(x * p_fx_x[None] + t * p_fx_t[None]) * ti.cos(y * p_fx_y[None]) + ti.sin(t * (p_fx_t[None]*0.6)) * p_fx_amp[None]
    elif p_preset[None] == 1:
        val = ti.sin(x * p_fx_x[None] * 2.0 + t * p_fx_t[None] * 1.5) * ti.cos(y * p_fx_y[None] * 0.5) + ti.sin(t * (p_fx_t[None]*1.0)) * p_fx_amp[None] * 1.5
    else:
        val = ti.cos(x * p_fx_x[None] - t * p_fx_t[None]) * ti.sin(y * p_fx_y[None] * 2.0) - ti.cos(t * p_fx_t[None]) * p_fx_amp[None]
    return val


@ti.func
def eq2_ti(x, y, t):
    val = 0.0
    if p_preset[None] == 0:
        val = ti.cos(x * p_fy_x[None] - t * p_fy_t[None]) * ti.sin(y * p_fy_y[None]) + ti.cos(t * (p_fy_t[None]*0.625)) * p_fy_amp[None]
    elif p_preset[None] == 1:
        val = ti.cos(x * p_fy_x[None] * 1.8 - t * p_fy_t[None] * 1.2) * ti.sin(y * p_fy_y[None] * 1.5) * p_fy_amp[None]
    else:
        val = ti.sin(x * p_fy_x[None] + y * p_fy_y[None] + t * p_fy_t[None]) * p_fy_amp[None]
    return val


@ti.func
def eq3_ti(x, y, t):
    dist = ti.sqrt((x - WIDTH/2)**2 + (y - HEIGHT/2)**2)
    val = 0.0
    if p_preset[None] == 0:
        val = ti.sin(dist * p_spiral_scale[None] - t * p_spiral_t[None]) * 0.8 + ti.cos(t * (p_spiral_t[None]*0.75))
    elif p_preset[None] == 1:
        val = ti.sin(dist * p_spiral_scale[None] * 4.0 - t * p_spiral_t[None] * 0.8) * 1.2
    else:
        val = ti.cos(dist * p_spiral_scale[None] * 2.5 + t * p_spiral_t[None])
    return val


@ti.func
def eq4_ti(x, y, t):
    val = 0.0
    if p_preset[None] == 0:
        val = ti.atan2(y - HEIGHT/2, x - WIDTH/2) * p_radial_strength[None] + ti.sin(t * (p_spiral_t[None]*0.5))
    elif p_preset[None] == 1:
        val = ti.atan2(y - HEIGHT/2, x - WIDTH/2) * p_radial_strength[None] * 2.0 + ti.cos(t * p_spiral_t[None])
    else:
        val = ti.sin(t * p_spiral_t[None]) * p_radial_strength[None] * 3.0
    return val


def get_color(value, mode="rainbow"):
    """Map scalar value to RGB color"""
    value = (value % 1.0 + 1.0) % 1.0  # Normalize to 0-1
    
    if mode == "rainbow":
        h = value
        s = 0.8
        v = 0.9
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        
        if h < 1/6:
            r, g, b = c, x, 0
        elif h < 2/6:
            r, g, b = x, c, 0
        elif h < 3/6:
            r, g, b = 0, c, x
        elif h < 4/6:
            r, g, b = 0, x, c
        elif h < 5/6:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (r + m, g + m, b + m)
    
    elif mode == "plasma":
        r = math.sin(value * 3.14159) ** 2
        g = math.sin((value + 0.33) * 3.14159) ** 2
        b = math.sin((value + 0.66) * 3.14159) ** 2
        return (r, g, b)
    
    elif mode == "ocean":
        r = 0.1 + value * 0.3
        g = 0.3 + value * 0.6
        b = 0.5 + value * 0.5
        return (r, g, b)
    
    elif mode == "fire":
        r = 1.0
        g = max(0, value * 2 - 1) if value > 0.5 else value * 2
        b = max(0, (1 - value) * 0.5)
        return (r, g, b)
    
    else:
        return (value, value, value)


@ti.kernel
def init_particles():
    """Initialize particles at random positions"""
    for i in particles_x:
        particles_x[i] = ti.random() * WIDTH
        particles_y[i] = ti.random() * HEIGHT
        particles_vx[i] = 0.0
        particles_vy[i] = 0.0
        particles_age[i] = 0.0
        particles_color[i] = ti.random()


def set_params_from_dict(d: dict):
    """Write a python dict of params into the taichi scalar fields"""
    if "p_fx_x" in d:
        p_fx_x[None] = float(d["p_fx_x"])
    if "p_fx_y" in d:
        p_fx_y[None] = float(d["p_fx_y"])
    if "p_fx_t" in d:
        p_fx_t[None] = float(d["p_fx_t"])
    if "p_fx_amp" in d:
        p_fx_amp[None] = float(d["p_fx_amp"])

    if "p_fy_x" in d:
        p_fy_x[None] = float(d["p_fy_x"])
    if "p_fy_y" in d:
        p_fy_y[None] = float(d["p_fy_y"])
    if "p_fy_t" in d:
        p_fy_t[None] = float(d["p_fy_t"])
    if "p_fy_amp" in d:
        p_fy_amp[None] = float(d["p_fy_amp"])

    if "p_spiral_scale" in d:
        p_spiral_scale[None] = float(d["p_spiral_scale"])
    if "p_spiral_t" in d:
        p_spiral_t[None] = float(d["p_spiral_t"])
    if "p_radial_strength" in d:
        p_radial_strength[None] = float(d["p_radial_strength"])
    if "p_global_speed" in d:
        p_global_speed[None] = float(d["p_global_speed"])
    if "preset" in d:
        p_preset[None] = int(d["preset"])


@ti.kernel
def update_particles(time: ti.f32):
    """Update particle positions based on equations"""
    for i in particles_x:
        x = particles_x[i]
        y = particles_y[i]
        
        # Calculate forces using kernel-callable equations (edit these `eq*_ti` funcs)
        fx = eq1_ti(x, y, time)
        fy = eq2_ti(x, y, time)

        # Fractal rotation influence (from eq3)
        dist = ti.sqrt((x - WIDTH/2)**2 + (y - HEIGHT/2)**2)
        spiral = eq3_ti(x, y, time)
        
        # Radial component (parameterized)
        dx = x - WIDTH/2
        dy = y - HEIGHT/2
        radial_strength = p_radial_strength[None]
        # Predeclare radial components so Taichi kernel sees them in all branches
        radial_x = 0.0
        radial_y = 0.0
        if dist > 0.1:
            a = ti.atan2(dy, dx)
            radial_x = (dx / dist) * a * radial_strength
            radial_y = (dy / dist) * a * radial_strength

        # Combine forces (include spiral influence)
        wobble_x = ti.sin(time * 0.1 + i * 0.001) * 0.3
        wobble_y = ti.cos(time * 0.1 + i * 0.001) * 0.3
        acceleration_x = fx + radial_x + spiral * 0.5 + wobble_x
        acceleration_y = fy + radial_y + spiral * 0.5 + wobble_y
        
        # Update velocity with stronger effect and damping
        particles_vx[i] = particles_vx[i] * 0.92 + acceleration_x * 0.3
        particles_vy[i] = particles_vy[i] * 0.92 + acceleration_y * 0.3
        
        # Update position scaled by global speed for visible changes
        particles_x[i] += particles_vx[i] * (1.0 + p_global_speed[None])
        particles_y[i] += particles_vy[i] * (1.0 + p_global_speed[None])
        
        # Wrap around edges
        if particles_x[i] < 0:
            particles_x[i] += WIDTH
        if particles_x[i] > WIDTH:
            particles_x[i] -= WIDTH
        if particles_y[i] < 0:
            particles_y[i] += HEIGHT
        if particles_y[i] > HEIGHT:
            particles_y[i] -= HEIGHT
        
        # Update age and color
        particles_age[i] += 1.0
        particles_color[i] = (particles_color[i] + 0.005) % 1.0


@ti.kernel
def clear_canvas():
    """Clear canvas with fade effect"""
    for i, j in canvas:
        canvas[i, j] *= 0.95


@ti.kernel
def render_particles():
    """Render particles to canvas"""
    for i in particles_x:
        x = ti.cast(particles_x[i], ti.i32)
        y = ti.cast(particles_y[i], ti.i32)
        
        if 0 <= x < WIDTH and 0 <= y < HEIGHT:
            # Simple additive blending
            canvas[x, y] += 1.0


def display_canvas(gui):
    """Convert Taichi field to displayable image"""
    img = canvas.to_numpy()
    # Normalize to 0-1 for display
    img = np.clip(img / 3.0, 0, 1)
    gui.set_image(img)


def main():
    """Main loop"""
    gui = ti.GUI("Liquid Fractal Life", res=(WIDTH, HEIGHT))
    init_particles()
    init_params()
    
    time = 0.0
    frame = 0
    
    print("\n=== LIQUID FRACTAL LIFE ===")
    print("Press 'r' to reset")
    print("Press 'q' or ESC to quit")
    print(f"Particles: {NUM_PARTICLES}")
    print(f"\nTip: Modify eq1, eq2, eq3, eq4 functions to change behavior!")
    print(f"Example changes to try:")
    print(f"  - Change multipliers: 0.01 → 0.02 (faster)")
    print(f"  - Change sin/cos combinations")
    print(f"  - Modify time scale: t * 0.5 → t * 2.0")
    print(f"  - Add more trigonometric functions")
    
    while gui.running:
        clear_canvas()
        # use global speed factor from params
        update_particles(time)
        render_particles()
        display_canvas(gui)
        
        # gui.show with a string tries to treat it as a filename; pass no arg to display
        gui.show()
        
        if gui.is_pressed('r'):
            init_particles()
            time = 0.0
            print("Reset!")
        
        if gui.is_pressed('q') or gui.is_pressed(ti.GUI.ESCAPE):
            break
        
        time += 0.01 * float(p_global_speed[None])
        frame += 1


if __name__ == "__main__":
    # Start a simple Tkinter control panel in a separate thread
    def tk_thread():
        root = tk.Tk()
        root.title("Liquid Fractal Controls")

        params = DEFAULT_PARAMS.copy()

        # create sliders for a few parameters
        sliders = {}
        def add_slider(key, label, low, high, row):
            tk.Label(parent_frame, text=label).grid(row=row, column=0, sticky='w')
            var = tk.DoubleVar(value=params[key])
            # update params in real-time when slider moves
            s = tk.Scale(parent_frame, variable=var, from_=low, to=high, resolution=0.0001, orient='horizontal', length=300, command=lambda val, k=key: set_params_from_dict({k: float(val)}))
            s.grid(row=row, column=1)
            sliders[key] = var

        parent_frame = tk.Frame(root)
        parent_frame.pack(padx=8, pady=8)

        add_slider('p_fx_x', 'fx_x', 0.0001, 0.05, 0)
        add_slider('p_fx_y', 'fx_y', 0.0001, 0.02, 1)
        add_slider('p_fx_t', 'fx_time', 0.01, 2.0, 2)
        add_slider('p_fx_amp', 'fx_amp', 0.0, 2.0, 3)

        add_slider('p_fy_x', 'fy_x', 0.0001, 0.05, 4)
        add_slider('p_fy_y', 'fy_y', 0.0001, 0.05, 5)
        add_slider('p_fy_t', 'fy_time', 0.01, 2.0, 6)
        add_slider('p_fy_amp', 'fy_amp', 0.0, 2.0, 7)

        add_slider('p_spiral_scale', 'spiral_scale', 0.00001, 0.01, 8)
        add_slider('p_spiral_t', 'spiral_time', 0.01, 2.0, 9)
        add_slider('p_radial_strength', 'radial_strength', 0.0, 1.0, 10)
        add_slider('p_global_speed', 'global_speed', 0.1, 5.0, 11)

        # Preset selector for very different equation sets
        preset_map = {"Default": 0, "Chaotic": 1, "Spiral": 2}
        var_preset = tk.StringVar(value="Default")
        tk.Label(parent_frame, text='Preset').grid(row=12, column=0, sticky='w')
        preset_menu = tk.OptionMenu(parent_frame, var_preset, *preset_map.keys())
        preset_menu.grid(row=12, column=1, sticky='w')
        # apply preset immediately when changed
        var_preset.trace_add('write', lambda *args: set_params_from_dict({'preset': preset_map[var_preset.get()]}))

        # Live equation editor (auto-recompile on edit) with helper toolbar and highlighting
        editor_label = tk.Label(root, text='Live Equation Editor (Taichi @ti.func code). Edits auto-recompile:')
        editor_label.pack(anchor='w', padx=8)

        # toolbar with common functions for easy insertion (like Desmos quick buttons)
        toolbar = tk.Frame(root)
        toolbar.pack(fill='x', padx=8)
        def insert_at_cursor(text_to_insert):
            editor.insert(tk.INSERT, text_to_insert)
            schedule_reload()
        for label, snippet in [
            ('sin()', 'ti.sin()'), ('cos()', 'ti.cos()'), ('tan()', 'ti.tan()'),
            ('sqrt()', 'ti.sqrt()'), ('atan2()', 'ti.atan2( , )'), ('pi', 'ti.pi'),
            ('x', 'x'), ('y', 'y'), ('t', 't'), ('**', '**')]:
            b = tk.Button(toolbar, text=label, width=6, command=lambda s=snippet: insert_at_cursor(s))
            b.pack(side='left', padx=2)

        editor_frame = tk.Frame(root)
        editor_frame.pack(padx=8, pady=4, fill='both', expand=True)

        # line numbers
        line_numbers = tk.Text(editor_frame, width=4, padx=4, takefocus=0, border=0, background='#f0f0f0', state='disabled')
        line_numbers.pack(side='left', fill='y')

        editor = tk.Text(editor_frame, height=12, width=60, wrap='none')
        editor.pack(side='left', fill='both', expand=True)
        scroll_y = tk.Scrollbar(editor_frame, command=lambda *args: (editor.yview(*args), line_numbers.yview(*args)))
        scroll_y.pack(side='right', fill='y')
        editor['yscrollcommand'] = lambda *args: (scroll_y.set(*args), line_numbers.yview_moveto(args[0]))

        editor_template = '''@ti.func
def eq1_ti(x, y, t):
    return ti.sin(x * p_fx_x[None] + t * p_fx_t[None]) * ti.cos(y * p_fx_y[None]) + ti.sin(t * (p_fx_t[None]*0.6)) * p_fx_amp[None]

@ti.func
def eq2_ti(x, y, t):
    return ti.cos(x * p_fy_x[None] - t * p_fy_t[None]) * ti.sin(y * p_fy_y[None]) + ti.cos(t * (p_fy_t[None]*0.625)) * p_fy_amp[None]

@ti.func
def eq3_ti(x, y, t):
    dist = ti.sqrt((x - WIDTH/2)**2 + (y - HEIGHT/2)**2)
    return ti.sin(dist * p_spiral_scale[None] - t * p_spiral_t[None]) * 0.8 + ti.cos(t * (p_spiral_t[None]*0.75))

@ti.func
def eq4_ti(x, y, t):
    return ti.atan2(y - HEIGHT/2, x - WIDTH/2) * p_radial_strength[None] + ti.sin(t * (p_spiral_t[None]*0.5))
'''
        editor.insert('1.0', editor_template)

        # basic syntax highlighting and line numbers
        import re
        HIGHLIGHT_PATTERNS = [
            (r"@ti\.func", 'kw'),
            (r"\bti\.[a-zA-Z_][a-zA-Z0-9_]*\b", 'func'),
            (r"\breturn\b|\bdef\b|\bimport\b", 'kw'),
            (r"\bWIDTH\b|\bHEIGHT\b|\bx\b|\by\b|\bt\b", 'var'),
            (r"#.*", 'comment'),
            (r"\b[0-9]+\.?[0-9]*\b", 'num'),
        ]
        editor.tag_config('kw', foreground='#007acc')
        editor.tag_config('func', foreground='#795e26')
        editor.tag_config('var', foreground='#001080')
        editor.tag_config('comment', foreground='#008000')
        editor.tag_config('num', foreground='#09885a')
        editor.tag_config('error', background='#ffdddd')

        def update_line_numbers():
            lines = int(editor.index('end-1c').split('.')[0])
            line_numbers.config(state='normal')
            line_numbers.delete('1.0', 'end')
            for i in range(1, lines+1):
                line_numbers.insert(f'{i}.0', f'{i}\n')
            line_numbers.config(state='disabled')

        def syntax_highlight():
            content = editor.get('1.0', 'end-1c')
            for tag in ['kw','func','var','comment','num']:
                editor.tag_remove(tag, '1.0', 'end')
            for pattern, tag in HIGHLIGHT_PATTERNS:
                for m in re.finditer(pattern, content):
                    start = '1.0 + %dc' % m.start()
                    end = '1.0 + %dc' % m.end()
                    editor.tag_add(tag, start, end)

        # Debounced live reload + highlight
        reload_timer = [None]
        def reload_equations():
            code = editor.get('1.0', 'end')
            try:
                exec(code, globals())
                status_label.config(text=f"Reloaded: {time.strftime('%H:%M:%S')}")
                editor.tag_remove('error', '1.0', 'end')
            except Exception as e:
                status_label.config(text=f"Reload error: {e}")
                # highlight entire editor background as error
                editor.tag_add('error', '1.0', 'end')
            reload_timer[0] = None

        def schedule_reload(event=None):
            update_line_numbers()
            syntax_highlight()
            if reload_timer[0]:
                root.after_cancel(reload_timer[0])
            reload_timer[0] = root.after(350, reload_equations)

        editor.bind('<KeyRelease>', schedule_reload)
        editor.bind('<MouseWheel>', lambda e: (line_numbers.yview_scroll(int(-1*(e.delta/120)), 'units')))

        status_label = tk.Label(root, text='Editor status: idle')
        status_label.pack(anchor='w', padx=8)

        def reset_defaults():
            for k, v in DEFAULT_PARAMS.items():
                if k in sliders:
                    sliders[k].set(DEFAULT_PARAMS[k])
                    set_params_from_dict({k: DEFAULT_PARAMS[k]})
            var_preset.set('Default')
            set_params_from_dict(DEFAULT_PARAMS)
            editor.delete('1.0', 'end')
            editor.insert('1.0', editor_template)
            schedule_reload()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text='Reset Defaults', command=reset_defaults).pack(side='left', padx=6)
        tk.Button(btn_frame, text='Reset Particles', command=lambda: init_particles()).pack(side='left', padx=6)

        # initial highlight
        schedule_reload()

        root.mainloop()

    thread = threading.Thread(target=tk_thread, daemon=True)
    thread.start()

    main()
