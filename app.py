"""
========================================================================
WEC Sensor Dashboard
========================================================================
A Plotly Dash app demonstrating sensor data analysis for a wave energy
converter (WEC) fleet. Built as an interview portfolio piece.

Tabs:
  0. Overview              - schematic + parameters
  1. Motion (Accel + IMU)  - wave-induced motion, gyro angular rates
  2. Strain + Dynamics     - resonance demo, FFT system identification
  3. Pressure              - 5 channels (hull/PTO/differential/depth/cabin)
  4. ADCP                  - current profile + GPS-corrected velocity
  5. Temperature           - 5 channels + bearing fault detection
  6. GPS                   - trajectory + geofencing + anomaly

Author: Youran Li
Stack:  Dash 2.x + Plotly + numpy/scipy
Style:  Restrained — minimal color, no emoji, monochrome-leaning.
"""

import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from scipy.ndimage import uniform_filter1d
import rainflow

try:
    from gravity_image import GRAVITY_B64
except ImportError:
    GRAVITY_B64 = None

import dash
from dash import dcc, html, Input, Output, callback


# ========================================================================
# STYLING
# ========================================================================
COLORS = {
    'bg':        '#FAFAFA',
    'card':      '#FFFFFF',
    'border':    '#D1D5DB',
    'text':      '#1F2937',
    'text_dim':  '#6B7280',
    'accent':    '#2563EB',
    'accent_2':  '#0F766E',
    'warning':   '#B45309',
    'alert':     '#B91C1C',
    'water':     '#B0CCE0',
    'plot_1':    '#1F2937',
    'plot_2':    '#2563EB',
    'plot_3':    '#6B7280',
}

FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

STYLES = {
    'app': {
        'fontFamily':   FONT,
        'backgroundColor': COLORS['bg'],
        'color':        COLORS['text'],
        'minHeight':    '100vh',
        'padding':      '0',
        'margin':       '0',
    },
    'header': {
        'borderBottom': f"1px solid {COLORS['border']}",
        'padding':      '24px 40px',
        'backgroundColor': COLORS['card'],
    },
    'title': {
        'fontSize':     '24px',
        'fontWeight':   '600',
        'margin':       '0',
        'color':        COLORS['text'],
    },
    'subtitle': {
        'fontSize':     '14px',
        'color':        COLORS['text_dim'],
        'marginTop':    '4px',
    },
    'tabs_container': {
        'padding':      '0 40px',
        'borderBottom': f"1px solid {COLORS['border']}",
        'backgroundColor': COLORS['card'],
    },
    'tab': {
        'padding':      '12px 20px',
        'fontWeight':   '500',
        'fontSize':     '14px',
        'border':       'none',
        'borderBottom': '2px solid transparent',
    },
    'tab_selected': {
        'borderBottom': f"2px solid {COLORS['accent']}",
        'color':        COLORS['accent'],
        'backgroundColor': COLORS['card'],
        'fontWeight':   '600',
    },
    'content': {
        'padding':      '32px 40px',
        'maxWidth':     '1400px',
        'margin':       '0 auto',
    },
    'tab_header': {
        'fontSize':     '20px',
        'fontWeight':   '600',
        'margin':       '0 0 8px 0',
        'color':        COLORS['text'],
    },
    'tab_subtitle': {
        'fontSize':     '14px',
        'color':        COLORS['text_dim'],
        'margin':       '0 0 24px 0',
        'lineHeight':   '1.5',
    },
    'two_col': {
        'display':      'grid',
        'gridTemplateColumns': '1fr 2fr',
        'gap':          '32px',
        'marginTop':    '24px',
    },
    'sidebar': {
        'backgroundColor': COLORS['card'],
        'border':       f"1px solid {COLORS['border']}",
        'borderRadius': '4px',
        'padding':      '20px',
    },
    'sidebar_section': {
        'marginBottom': '20px',
    },
    'sidebar_h': {
        'fontSize':     '13px',
        'fontWeight':   '600',
        'color':        COLORS['text_dim'],
        'textTransform': 'uppercase',
        'letterSpacing': '0.05em',
        'margin':       '0 0 8px 0',
    },
    'sidebar_body': {
        'fontSize':     '14px',
        'lineHeight':   '1.6',
        'color':        COLORS['text'],
    },
    'param_table': {
        'fontSize':     '13px',
        'width':        '100%',
        'borderCollapse': 'collapse',
    },
    'param_label': {
        'padding':      '6px 8px 6px 0',
        'color':        COLORS['text_dim'],
        'fontFamily':   "'SF Mono', Menlo, monospace",
        'fontSize':     '12px',
    },
    'param_value': {
        'padding':      '6px 0',
        'fontFamily':   "'SF Mono', Menlo, monospace",
        'fontSize':     '12px',
        'fontWeight':   '500',
        'color':        COLORS['text'],
    },
    'plots': {
        'display':      'flex',
        'flexDirection': 'column',
        'gap':          '16px',
    },
}


def plot_layout(**kwargs):
    """Standard Plotly layout for all figures -- consistent visual style."""
    base = dict(
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(family=FONT, size=12, color=COLORS['text']),
        margin=dict(l=60, r=20, t=40, b=50),
        xaxis=dict(gridcolor='#F0F0F0', linecolor=COLORS['border'],
                   zerolinecolor='#E5E7EB'),
        yaxis=dict(gridcolor='#F0F0F0', linecolor=COLORS['border'],
                   zerolinecolor='#E5E7EB'),
        showlegend=True,
        legend=dict(font=dict(size=11), borderwidth=0, bgcolor='rgba(255,255,255,0.8)'),
        hovermode='x unified',
    )
    base.update(kwargs)
    return base


# ========================================================================
# REUSABLE UI HELPERS (text outputs, badges)
# ========================================================================
def result_block(title, content_lines):
    """
    Monospace-formatted text output block, like terminal print output.
    content_lines: list of strings (each becomes a line) or a single string.
    """
    if isinstance(content_lines, list):
        text = '\n'.join(content_lines)
    else:
        text = content_lines
    return html.Div([
        html.Div(title, style={
            'fontSize':     '11px',
            'fontWeight':   '600',
            'color':        COLORS['text_dim'],
            'textTransform': 'uppercase',
            'letterSpacing': '0.05em',
            'marginBottom': '6px',
        }),
        html.Pre(text, style={
            'fontFamily':   "'SF Mono', Menlo, monospace",
            'fontSize':     '12px',
            'color':        COLORS['text'],
            'backgroundColor': '#F9FAFB',
            'border':       f"1px solid {COLORS['border']}",
            'borderRadius': '4px',
            'padding':      '14px',
            'margin':       '0',
            'overflowX':    'auto',
            'lineHeight':   '1.55',
            'whiteSpace':   'pre-wrap',
        }),
    ], style={'marginBottom': '16px'})


def verdict_badge(label, passed, note=''):
    """Single PASS/FAIL badge for a criterion."""
    color = COLORS['accent_2'] if passed else COLORS['alert']
    status_text = 'PASS' if passed else 'FAIL'
    return html.Div([
        html.Span(label, style={
            'fontSize':     '13px',
            'color':        COLORS['text'],
            'marginRight':  '12px',
        }),
        html.Span(status_text, style={
            'display':      'inline-block',
            'padding':      '3px 10px',
            'borderRadius': '3px',
            'backgroundColor': color,
            'color':        'white',
            'fontSize':     '11px',
            'fontWeight':   '700',
            'letterSpacing': '0.05em',
        }),
        html.Span(f' {note}' if note else '', style={
            'fontSize':     '12px',
            'color':        COLORS['text_dim'],
            'marginLeft':   '8px',
        }),
    ], style={'marginBottom': '6px'})


def overall_verdict(passed):
    """Big colored banner for overall design verdict."""
    if passed:
        bg = COLORS['accent_2']
        text = 'DESIGN PASSES — the float is expected to reach 20-year design life.'
    else:
        bg = COLORS['alert']
        text = 'DESIGN FAILS — structure not expected to reach 20-year design life.'
    return html.Div(text, style={
        'padding':      '14px 20px',
        'borderRadius': '4px',
        'backgroundColor': bg,
        'color':        'white',
        'fontSize':     '15px',
        'fontWeight':   '600',
        'marginTop':    '12px',
        'marginBottom': '20px',
    })


def section_header(text):
    """Section header within a tab (smaller than tab_header)."""
    return html.H3(text, style={
        'fontSize':     '16px',
        'fontWeight':   '600',
        'color':        COLORS['text'],
        'margin':       '32px 0 12px 0',
        'paddingBottom': '6px',
        'borderBottom': f"1px solid {COLORS['border']}",
    })


def part_header(text):
    """Big top-level part header — bigger and more visible than section_header.
    Use for grouping sections into PART 1 / PART 2 type structure."""
    return html.Div([
        html.Div(text, style={
            'fontSize':     '20px',
            'fontWeight':   '700',
            'color':        COLORS['accent'],
            'padding':      '14px 0 10px 0',
            'borderTop':    f"3px solid {COLORS['accent']}",
            'borderBottom': f"1px solid {COLORS['border']}",
            'letterSpacing': '0.02em',
        }),
    ], style={'margin': '40px 0 18px 0'})


def fmt_value(v, decimals=2):
    """Format a numeric value for human-readable display.

    Rules (no scientific-notation `e±N` ever):
      * Zero → '0'
      * Integers ≥ 1000 → comma-separated (e.g. 10,000,000)
      * |v| in [1e-3, 1e6) → plain decimal with adaptive precision
        (so 0.14 shows as '0.14' and 0.0014 shows as '0.0014',
        not as '0.00')
      * |v| < 1e-3 or |v| ≥ 1e6 → Unicode scientific notation
        with × 10ⁿ superscript (e.g. 7.98 × 10⁻⁶)
    """
    if v is None:
        return '—'
    try:
        v_float = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v_float == 0:
        return '0'
    if not np.isfinite(v_float):
        return '∞' if v_float > 0 else '-∞'
    abs_v = abs(v_float)
    # Integers ≥ 1000 → commas
    if v_float == int(v_float) and abs_v >= 1000:
        return f'{int(v_float):,}'
    # Plain decimal range: choose precision so we always show
    # at least `decimals` significant figures.
    if 1e-3 <= abs_v < 1e6:
        if abs_v >= 1:
            d = decimals
        else:
            # need enough decimals to show `decimals` sig figs
            d = decimals + int(-np.floor(np.log10(abs_v)))
        s = f'{v_float:.{d}f}'
        if '.' in s:
            s = s.rstrip('0').rstrip('.')
        return s if s else '0'
    # Unicode scientific notation
    exp = int(np.floor(np.log10(abs_v)))
    mant = v_float / (10 ** exp)
    sup_map = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    return f'{mant:.{decimals}f} × 10{str(exp).translate(sup_map)}'


def styled_table(headers, rows, highlight_row=None, footer_note=None):
    """
    Render a modern styled HTML table.

    headers: list of column header strings
    rows: list of lists, each inner list is one row's cell values
    highlight_row: optional index of a row to emphasize (light red background)
    footer_note: optional small italic note below the table
    """
    header_style = {
        'padding':       '10px 14px',
        'fontSize':      '11px',
        'fontWeight':    '600',
        'color':         COLORS['text_dim'],
        'textTransform': 'uppercase',
        'letterSpacing': '0.06em',
        'borderBottom':  f"2px solid {COLORS['border']}",
        'textAlign':     'left',
        'backgroundColor': '#F9FAFB',
        'fontFamily':    FONT,
    }
    cell_style_base = {
        'padding':      '10px 14px',
        'fontSize':     '13px',
        'color':        COLORS['text'],
        'borderBottom': f"1px solid {COLORS['border']}",
        'fontFamily':   FONT,
    }
    rows_html = []
    for i, row in enumerate(rows):
        row_style = {}
        cell_style = dict(cell_style_base)
        if i == highlight_row:
            row_style = {'backgroundColor': '#FEF2F2'}
            cell_style = {**cell_style_base, 'fontWeight': '600'}
        rows_html.append(html.Tr([
            html.Td(c, style=cell_style) for c in row
        ], style=row_style))

    parts = [html.Table([
        html.Thead(html.Tr([html.Th(h, style=header_style) for h in headers])),
        html.Tbody(rows_html),
    ], style={
        'width':           '100%',
        'borderCollapse':  'collapse',
        'marginTop':       '8px',
        'marginBottom':    '8px',
        'backgroundColor': '#FFFFFF',
        'border':          f"1px solid {COLORS['border']}",
        'borderRadius':    '4px',
        'overflow':        'hidden',
    })]
    if footer_note:
        parts.append(html.Div(footer_note, style={
            'fontSize':   '12px',
            'color':      COLORS['text_dim'],
            'fontStyle':  'italic',
            'margin':     '4px 4px 16px 4px',
        }))
    return html.Div(parts)


def explanation_block(content):
    """A reader-friendly explanation block — like talking to the user."""
    return html.Div(content, style={
        'fontSize':     '13.5px',
        'lineHeight':   '1.65',
        'color':        COLORS['text'],
        'margin':       '12px 0 16px 0',
        'paddingLeft':  '14px',
        'borderLeft':   f"3px solid {COLORS['accent']}",
    })


def latex_equation(latex_text):
    """
    Display a LaTeX equation using MathJax inline syntax with display=true.
    Inline math (`$...$`) is known to render in this Dash setup; we wrap it
    in a centered styled box for a display-equation feel.
    Pass the LaTeX content WITHOUT the $ delimiters.
    Example: latex_equation(r'H(\\omega) = \\frac{1}{k - m\\omega^2 + ic\\omega}')
    """
    return html.Div([
        dcc.Markdown(
            f'$\\displaystyle {latex_text}$',
            mathjax=True,
            style={'fontSize': '17px', 'margin': '0'},
        ),
    ], style={
        'textAlign':       'center',
        'padding':         '14px 20px',
        'margin':          '12px 0',
        'backgroundColor': '#EEF2F7',
        'border':          f"1px solid {COLORS['border']}",
        'borderRadius':    '4px',
    })


def expandable_note(title, content):
    """
    Native HTML5 collapsible section. Closed by default — user clicks the
    summary line to expand. Lightweight and JS-free.

    title: short label shown next to the + icon (e.g. "Reading the table")
    content: any Dash component or string to show when expanded
    """
    return html.Details([
        html.Summary([
            html.Span('+  ', style={
                'fontWeight':  '700',
                'color':       COLORS['accent'],
                'fontFamily':  'monospace',
            }),
            html.Span(title, style={
                'fontSize':    '13px',
                'fontWeight':  '600',
                'color':       COLORS['text'],
            }),
        ], style={
            'cursor':          'pointer',
            'padding':         '10px 14px',
            'backgroundColor': '#F9FAFB',
            'border':          f"1px solid {COLORS['border']}",
            'borderRadius':    '4px',
            'userSelect':      'none',
            'listStyle':       'none',
        }),
        html.Div(content, style={
            'padding':         '14px 18px',
            'fontSize':        '13.5px',
            'lineHeight':      '1.65',
            'color':           COLORS['text'],
            'backgroundColor': '#FFFFFF',
            'border':          f"1px solid {COLORS['border']}",
            'borderTop':       'none',
            'borderRadius':    '0 0 4px 4px',
            'marginTop':       '-1px',
        }),
    ], style={'margin': '8px 0 16px 0'})


def physics_block(markdown_text):
    """Render a math/physics explanation block with LaTeX support."""
    return html.Div(
        dcc.Markdown(markdown_text, mathjax=True, style={
            'fontSize':   '13px',
            'lineHeight': '1.6',
            'color':      COLORS['text'],
        }),
        style={
            'backgroundColor': '#F5F7FA',
            'border':       f"1px solid {COLORS['border']}",
            'borderRadius': '4px',
            'padding':      '16px 20px',
            'margin':       '12px 0',
        },
    )


def equation_block(equation_text):
    """
    Display a math equation using Unicode + styled HTML. Reliable —
    does not depend on MathJax loading (which has been inconsistent in Dash).

    Pass Unicode text directly, e.g.:
        equation_block("m·ẍ(t) + c·ẋ(t) + k·x(t) = F(t)")
    """
    return html.Div(equation_text, style={
        'fontFamily':   "'Cambria Math', 'Latin Modern Math', 'STIX Two Math', "
                        "'Times New Roman', serif",
        'fontStyle':    'italic',
        'fontSize':     '17px',
        'textAlign':    'center',
        'padding':      '14px 20px',
        'margin':       '12px 0',
        'backgroundColor': '#EEF2F7',
        'border':       f"1px solid {COLORS['border']}",
        'borderRadius': '4px',
        'color':        COLORS['text'],
        'letterSpacing': '0.02em',
        'lineHeight':   '1.5',
    })


def physics_explanation(title, body_text, equations=None):
    """
    Combined physics block: bold title (MathJax-aware) + prose explanation
    (with inline MathJax support, e.g. $\\omega_n$) + optional equations.
    """
    parts = [
        dcc.Markdown(f"**{title}**", mathjax=True, style={
            'fontSize':     '14px',
            'color':        COLORS['text'],
            'marginBottom': '4px',
        }),
        dcc.Markdown(body_text, mathjax=True, style={
            'fontSize':     '13.5px',
            'lineHeight':   '1.65',
            'color':        COLORS['text'],
            'marginBottom': '4px',
        }),
    ]
    if equations:
        for eq in equations:
            parts.append(equation_block(eq))
    return html.Div(parts, style={
        'backgroundColor': '#F9FAFB',
        'border':       f"1px solid {COLORS['border']}",
        'borderRadius': '4px',
        'padding':      '16px 20px',
        'margin':       '12px 0',
    })


# ========================================================================
# DIAGRAM BUILDERS (used in Overview and Dynamics tabs)
# ========================================================================
def make_6dof_figure():
    """6 degrees-of-freedom diagram — ship-shaped icons for visual clarity.
    Translates the 6DOF concept into something more intuitive than rectangles.
    Bottom row labels include the Euler-angle symbols (φ, θ, ψ).
    """
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'SURGE — slide forward/back',
            'SWAY — slide left/right',
            'HEAVE — bob up/down',
            'ROLL  φ  — tilt side-to-side',
            'PITCH  θ  — tilt nose up/down',
            'YAW  ψ  — turn left/right',
        ),
        horizontal_spacing=0.06, vertical_spacing=0.20,
    )

    hull_color = '#8E8A7B'
    cabin_color = '#6B6B6B'
    edge = COLORS['text']
    arrow_color = COLORS['accent']
    rot_color = '#A37B1E'
    water_color = COLORS['water']

    def add_ship_side(row, col, x_shift=0, tilt_deg=0):
        """Ship side view: hull (trapezoid with bow) + cabin on top."""
        # Hull profile points (boat shape, side view)
        # Going clockwise from stern-bottom: stern-bottom, stern-top, bow-top, bow-tip, bow-bottom
        import math
        ang = math.radians(tilt_deg)
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        # Original points (centered at 0)
        pts = [(-0.5, -0.15), (-0.5, 0.05), (0.35, 0.05), (0.55, -0.05), (0.5, -0.15)]
        rot = [(x_shift + p[0]*cos_a - p[1]*sin_a,
                 p[0]*sin_a + p[1]*cos_a) for p in pts]
        path = 'M ' + ' L '.join(f"{x:.3f} {y:.3f}" for x, y in rot) + ' Z'
        fig.add_shape(type='path', path=path,
                       fillcolor=hull_color, line=dict(color=edge, width=1.5),
                       row=row, col=col)
        # Cabin
        cab_pts = [(-0.18, 0.05), (-0.18, 0.22), (0.15, 0.22), (0.15, 0.05)]
        rot_c = [(x_shift + p[0]*cos_a - p[1]*sin_a,
                  p[0]*sin_a + p[1]*cos_a) for p in cab_pts]
        path_c = 'M ' + ' L '.join(f"{x:.3f} {y:.3f}" for x, y in rot_c) + ' Z'
        fig.add_shape(type='path', path=path_c,
                       fillcolor=cabin_color, line=dict(color=edge, width=1.5),
                       row=row, col=col)

    def add_ship_top(row, col, x_shift=0, rot_deg=0):
        """Ship top view: bullet/oval shape with pointed bow."""
        import math
        ang = math.radians(rot_deg)
        cos_a, sin_a = math.cos(ang), math.sin(ang)
        # Bullet shape (oval body + pointed tip at right = bow)
        pts = []
        # left half (back) - half oval
        for theta in [math.pi/2 + i*math.pi/12 for i in range(13)]:
            pts.append((-0.2 + 0.3*math.cos(theta), 0.15*math.sin(theta)))
        # bow point at right
        pts.append((0.5, 0))
        rot = [(x_shift + p[0]*cos_a - p[1]*sin_a,
                 p[0]*sin_a + p[1]*cos_a) for p in pts]
        path = 'M ' + ' L '.join(f"{x:.3f} {y:.3f}" for x, y in rot) + ' Z'
        fig.add_shape(type='path', path=path,
                       fillcolor=hull_color, line=dict(color=edge, width=1.5),
                       row=row, col=col)
        # Cabin (small rect on top)
        cab_pts = [(-0.12, -0.06), (-0.12, 0.06), (0.05, 0.06), (0.05, -0.06)]
        rot_c = [(x_shift + p[0]*cos_a - p[1]*sin_a,
                  p[0]*sin_a + p[1]*cos_a) for p in cab_pts]
        path_c = 'M ' + ' L '.join(f"{x:.3f} {y:.3f}" for x, y in rot_c) + ' Z'
        fig.add_shape(type='path', path=path_c,
                       fillcolor=cabin_color, line=dict(color=edge, width=1.5),
                       row=row, col=col)

    def add_ship_front(row, col):
        """Front view of ship: roughly trapezoidal hull cross-section + cabin."""
        # Hull cross-section (wider at top, narrow at bottom)
        path = 'M -0.35 0 L -0.45 0.15 L 0.45 0.15 L 0.35 0 Z'
        fig.add_shape(type='path', path=path,
                       fillcolor=hull_color, line=dict(color=edge, width=1.5),
                       row=row, col=col)
        # Cabin (small box on top)
        fig.add_shape(type='rect', x0=-0.15, y0=0.15, x1=0.15, y1=0.32,
                       fillcolor=cabin_color, line=dict(color=edge, width=1.5),
                       row=row, col=col)

    def add_water_line(row, col):
        fig.add_shape(type='line', x0=-1, y0=-0.2, x1=1, y1=-0.2,
                       line=dict(color=water_color, width=2), row=row, col=col)

    def add_double_arrow(x0, x1, y, row, col, color=arrow_color):
        """Horizontal double-headed arrow at height y from x0 to x1."""
        xref = f'x{(row-1)*3 + col}'
        yref = f'y{(row-1)*3 + col}'
        fig.add_annotation(x=x1, y=y, ax=x0, ay=y,
                            xref=xref, yref=yref, axref=xref, ayref=yref,
                            arrowhead=3, arrowsize=1.0, arrowwidth=2.5,
                            arrowcolor=color, showarrow=True)
        fig.add_annotation(x=x0, y=y, ax=x1, ay=y,
                            xref=xref, yref=yref, axref=xref, ayref=yref,
                            arrowhead=3, arrowsize=1.0, arrowwidth=2.5,
                            arrowcolor=color, showarrow=True)

    def add_vert_arrow(x, y0, y1, row, col, color=arrow_color):
        xref = f'x{(row-1)*3 + col}'
        yref = f'y{(row-1)*3 + col}'
        fig.add_annotation(x=x, y=y1, ax=x, ay=y0,
                            xref=xref, yref=yref, axref=xref, ayref=yref,
                            arrowhead=3, arrowsize=1.0, arrowwidth=2.5,
                            arrowcolor=color, showarrow=True)
        fig.add_annotation(x=x, y=y0, ax=x, ay=y1,
                            xref=xref, yref=yref, axref=xref, ayref=yref,
                            arrowhead=3, arrowsize=1.0, arrowwidth=2.5,
                            arrowcolor=color, showarrow=True)

    # === ROW 1: TRANSLATIONS ===
    # SURGE — side view, horizontal arrow (forward/back)
    add_water_line(1, 1)
    add_ship_side(1, 1, x_shift=0)
    add_double_arrow(-0.7, 0.7, -0.5, 1, 1)

    # SWAY — top view, horizontal arrow (left/right)
    add_ship_top(1, 2, x_shift=0)
    add_double_arrow(-0.7, 0.7, -0.5, 1, 2)

    # HEAVE — side view, vertical arrow
    add_water_line(1, 3)
    add_ship_side(1, 3, x_shift=0)
    add_vert_arrow(0.75, -0.5, 0.7, 1, 3)

    # === ROW 2: ROTATIONS ===
    # ROLL — front view, curved arrow indicating side-to-side tilt
    add_ship_front(2, 1)
    fig.add_annotation(x=0, y=-0.6, text='↻  around forward (longitudinal) axis',
                        showarrow=False, font=dict(size=11, color=rot_color),
                        xref='x4', yref='y4')

    # PITCH — side view, tilted to show nose up
    add_water_line(2, 2)
    add_ship_side(2, 2, x_shift=0, tilt_deg=15)
    fig.add_annotation(x=0, y=-0.6, text='↻  around side (transverse) axis',
                        showarrow=False, font=dict(size=11, color=rot_color),
                        xref='x5', yref='y5')

    # YAW — top view, rotated to show heading change
    add_ship_top(2, 3, x_shift=0, rot_deg=20)
    fig.add_annotation(x=0, y=-0.6, text='↻  around vertical axis',
                        showarrow=False, font=dict(size=11, color=rot_color),
                        xref='x6', yref='y6')

    fig.update_xaxes(visible=False, range=[-1, 1])
    fig.update_yaxes(visible=False, range=[-0.85, 0.55], scaleanchor='x',
                      scaleratio=1)
    fig.update_layout(plot_layout(
        height=400, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    ))
    # Subplot titles styling
    for ann in fig['layout']['annotations']:
        txt = getattr(ann, 'text', '') or ''
        if any(dof in txt for dof in
                ['SURGE', 'SWAY', 'HEAVE', 'ROLL', 'PITCH', 'YAW']):
            ann['font'] = dict(size=11, color=COLORS['text'], family=FONT)
    return fig


def make_neck_bending_figure():
    """Neck bending diagram — at rest vs during wave."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('At rest — no bending, no strain',
                         'During wave — neck bends → strain at weld'),
        horizontal_spacing=0.15,
    )

    # Common elements
    body_color = '#8E8A7B'
    neck_color = '#6B6B6B'
    weld_color = COLORS['warning']
    water_color = COLORS['water']

    # LEFT PANEL: at rest (straight)
    # Water
    fig.add_shape(type='rect', x0=-2, y0=-5, x1=2, y1=0,
                   fillcolor=water_color, opacity=0.3, line_width=0, row=1, col=1)
    # Float (top)
    fig.add_shape(type='rect', x0=-0.8, y0=0, x1=0.8, y1=1.2,
                   fillcolor=body_color, line=dict(color=COLORS['text'], width=1.5),
                   row=1, col=1)
    # Neck (straight)
    fig.add_shape(type='rect', x0=-0.18, y0=-4, x1=0.18, y1=0,
                   fillcolor=neck_color, line=dict(color=COLORS['text'], width=1.5),
                   row=1, col=1)
    # Bottom (turbine)
    fig.add_shape(type='rect', x0=-0.6, y0=-4.6, x1=0.6, y1=-4,
                   fillcolor='#404040', line=dict(color=COLORS['text'], width=1.5),
                   row=1, col=1)
    # Weld markers (yellow)
    fig.add_shape(type='line', x0=-0.18, y0=-0.05, x1=0.18, y1=-0.05,
                   line=dict(color=weld_color, width=4), row=1, col=1)
    fig.add_shape(type='line', x0=-0.18, y0=-3.95, x1=0.18, y1=-3.95,
                   line=dict(color=weld_color, width=4), row=1, col=1)
    fig.add_annotation(x=0.4, y=-0.05, text=' weld (gauge here)',
                        showarrow=False, font=dict(size=11, color=COLORS['text']),
                        xanchor='left', row=1, col=1)

    # RIGHT PANEL: bent (force from left)
    # Water
    fig.add_shape(type='rect', x0=-2, y0=-5, x1=2, y1=0,
                   fillcolor=water_color, opacity=0.3, line_width=0, row=1, col=2)
    # Float (top, shifted right)
    fig.add_shape(type='rect', x0=-0.4, y0=0, x1=1.2, y1=1.2,
                   fillcolor=body_color, line=dict(color=COLORS['text'], width=1.5),
                   row=1, col=2)
    # Curved neck — use a series of small rectangles forming a bow
    n_seg = 14
    for i in range(n_seg):
        y0 = -4 + i * (4 / n_seg)
        y1 = -4 + (i+1) * (4 / n_seg)
        x_offset = 0.4 * (1 - (1 - i / n_seg)**2)  # parabolic bend
        fig.add_shape(type='rect', x0=x_offset - 0.18, y0=y0,
                       x1=x_offset + 0.18, y1=y1,
                       fillcolor=neck_color,
                       line=dict(color=COLORS['text'], width=0.5),
                       row=1, col=2)
    # Bottom (turbine, fixed)
    fig.add_shape(type='rect', x0=-0.6, y0=-4.6, x1=0.6, y1=-4,
                   fillcolor='#404040', line=dict(color=COLORS['text'], width=1.5),
                   row=1, col=2)
    # Wave force arrow
    fig.add_annotation(x=-0.4, y=0.6, ax=-1.5, ay=0.6,
                        xref='x2', yref='y2', axref='x2', ayref='y2',
                        arrowhead=3, arrowsize=1.5, arrowwidth=3,
                        arrowcolor=COLORS['alert'],
                        text='wave force', font=dict(size=11, color=COLORS['alert']),
                        showarrow=True, xanchor='right')
    # Strain labels
    fig.add_annotation(x=-0.4, y=-0.5, text='TENSION  (+ε)',
                        showarrow=False,
                        font=dict(size=10, color=COLORS['accent_2']),
                        xanchor='right', row=1, col=2)
    fig.add_annotation(x=0.4, y=-0.5, text='COMPRESSION (−ε)',
                        showarrow=False,
                        font=dict(size=10, color=COLORS['warning']),
                        xanchor='left', row=1, col=2)
    fig.add_shape(type='line', x0=-0.18, y0=-0.05, x1=0.18, y1=-0.05,
                   line=dict(color=weld_color, width=4), row=1, col=2)

    fig.update_xaxes(visible=False, range=[-2, 2])
    fig.update_yaxes(visible=False, range=[-5, 2], scaleanchor='x', scaleratio=1)
    fig.update_layout(plot_layout(
        height=420, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    ))
    for ann in fig['layout']['annotations']:
        txt = getattr(ann, 'text', '') or ''
        if any(t in txt for t in ['At rest', 'During wave']):
            ann['font'] = dict(size=12, color=COLORS['text'])
    return fig


def make_wave_direction_figure(theta_deg=30.0):
    """Top-down view of the float showing wave incidence angle θ
    relative to the body x-axis."""
    import math
    fig = go.Figure()
    theta = math.radians(theta_deg)

    # Background sea
    fig.add_shape(type='rect', x0=-3, y0=-3, x1=3, y1=3,
                   fillcolor=COLORS['water'], opacity=0.18, line_width=0,
                   layer='below')

    # Wave crests as parallel lines, perpendicular to the propagation direction
    # Wave direction (unit vector): (cos θ, sin θ)
    # Wave crests are perpendicular: (-sin θ, cos θ) direction
    wave_dir = (math.cos(theta), math.sin(theta))
    crest_dir = (-math.sin(theta), math.cos(theta))
    for d_offset in [-2.5, -2.0, -1.5, -1.0, -0.5, 0.0]:
        cx = d_offset * wave_dir[0]
        cy = d_offset * wave_dir[1]
        x0 = cx - 2.4 * crest_dir[0]
        y0 = cy - 2.4 * crest_dir[1]
        x1 = cx + 2.4 * crest_dir[0]
        y1 = cy + 2.4 * crest_dir[1]
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                       line=dict(color=COLORS['water'], width=1.5),
                       opacity=0.55, layer='below')

    # Float (top-down bullet shape, bow pointing right = body x-axis)
    bow_pts = []
    for ang_deg in range(90, 271, 15):
        a = math.radians(ang_deg)
        bow_pts.append((-0.4 + 0.5 * math.cos(a), 0.45 * math.sin(a)))
    bow_pts.append((0.7, 0.0))
    path = 'M ' + ' L '.join(f"{x:.3f} {y:.3f}" for x, y in bow_pts) + ' Z'
    fig.add_shape(type='path', path=path,
                   fillcolor='#8E8A7B', line=dict(color=COLORS['text'], width=1.5))

    # Body x-axis arrow (forward)
    fig.add_annotation(x=2.0, y=0, ax=0.8, ay=0,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2,
                        arrowcolor=COLORS['text'])
    fig.add_annotation(x=2.05, y=0.15, text='body  x',
                        showarrow=False,
                        font=dict(size=12, color=COLORS['text'], family=FONT),
                        xanchor='left')
    # Body y-axis arrow (left)
    fig.add_annotation(x=0, y=1.6, ax=0, ay=0.5,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2,
                        arrowcolor=COLORS['text'])
    fig.add_annotation(x=0.15, y=1.65, text='body  y',
                        showarrow=False,
                        font=dict(size=12, color=COLORS['text'], family=FONT),
                        xanchor='left')

    # Wave direction arrow (incoming from upper-left, pointing toward float)
    src_x = -2.4 * math.cos(theta)
    src_y = -2.4 * math.sin(theta)
    # Arrow points TOWARD the float
    fig.add_annotation(x=-0.5 * math.cos(theta), y=-0.5 * math.sin(theta),
                        ax=src_x, ay=src_y,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=2.4,
                        arrowcolor=COLORS['accent'])
    fig.add_annotation(
        x=src_x*0.7 - 0.15*math.sin(theta),
        y=src_y*0.7 + 0.15*math.cos(theta),
        text='wave direction',
        showarrow=False,
        font=dict(size=12, color=COLORS['accent'], family=FONT, weight='bold'),
    )

    # Arc showing θ angle (between body-x axis and the line from float to wave source)
    arc_r = 0.95
    n_arc = 30
    # Arc goes from angle 0 (along +x) to angle (theta + 180°) — the line from float
    # pointing back to the wave source. Wait, actually we want θ between body x
    # and the LINE TOWARDS THE WAVE SOURCE (incoming wave direction).
    # The wave is coming from angle (180° + theta_deg) direction. The "wave
    # incidence angle" θ that we report is measured from body x to the wave's
    # incoming direction line — same as theta_deg if measured from -x.
    # For clarity: just draw arc from +x to the wave-source ray (angle 180° + θ).
    arc_start = 0  # body +x
    arc_end = math.pi + theta
    arc_angles = [arc_start + (arc_end - arc_start) * i / n_arc for i in range(n_arc + 1)]
    arc_x = [arc_r * math.cos(a) for a in arc_angles]
    arc_y = [arc_r * math.sin(a) for a in arc_angles]
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y, mode='lines',
        line=dict(color=COLORS['accent'], width=2, dash='dot'),
        showlegend=False, hoverinfo='skip',
    ))
    # θ label at the middle of the arc
    mid_angle = (arc_start + arc_end) / 2
    fig.add_annotation(
        x=(arc_r + 0.25) * math.cos(mid_angle),
        y=(arc_r + 0.25) * math.sin(mid_angle),
        text=f'θ = {theta_deg:.0f}°',
        showarrow=False,
        font=dict(size=14, color=COLORS['accent'], family=FONT, weight='bold'),
    )

    fig.update_xaxes(range=[-3, 3], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-2.6, 2.6], visible=False, fixedrange=True,
                     scaleanchor='x', scaleratio=1)
    fig.update_layout(plot_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text='Top-down view — wave arriving at the float at angle θ '
                         'from the body x-axis',
                    font=dict(size=13)),
        showlegend=False,
    ))
    return fig


def make_gravity_projection_figure():
    """Side view showing how gravity projects onto a tilted float's body axes.
    When the float pitches by angle θ, gravity (world-vertical) splits into
    components along body-x (= −g sin θ) and body-z (= g cos θ).
    """
    import math
    fig = go.Figure()
    pitch_deg = 25.0
    pitch = math.radians(pitch_deg)
    g_len = 1.4
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    # Background
    fig.add_shape(type='rect', x0=-2.2, y0=-2.2, x1=2.2, y1=1.4,
                   fillcolor='#F7F7F7', opacity=1.0, line_width=0,
                   layer='below')

    # Float side view (tilted by pitch)
    pts = [(-0.7, -0.18), (-0.7, 0.05), (0.5, 0.05), (0.75, -0.07), (0.7, -0.18)]
    rot = [(p[0]*cos_p - p[1]*sin_p, p[0]*sin_p + p[1]*cos_p) for p in pts]
    path = 'M ' + ' L '.join(f"{x:.3f} {y:.3f}" for x, y in rot) + ' Z'
    fig.add_shape(type='path', path=path,
                   fillcolor='#8E8A7B', line=dict(color=COLORS['text'], width=1.5))
    # Cabin
    cab = [(-0.22, 0.05), (-0.22, 0.28), (0.20, 0.28), (0.20, 0.05)]
    rot_c = [(p[0]*cos_p - p[1]*sin_p, p[0]*sin_p + p[1]*cos_p) for p in cab]
    path_c = 'M ' + ' L '.join(f"{x:.3f} {y:.3f}" for x, y in rot_c) + ' Z'
    fig.add_shape(type='path', path=path_c,
                   fillcolor='#6B6B6B', line=dict(color=COLORS['text'], width=1.5))

    # Gravity vector (world-vertical, straight down from above the float)
    g_tip_x = 0
    g_tip_y = -g_len
    fig.add_annotation(x=g_tip_x, y=g_tip_y, ax=0, ay=0.6,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=3, arrowsize=1.5, arrowwidth=2.4,
                        arrowcolor=COLORS['alert'])
    fig.add_annotation(x=-0.18, y=-0.7, text='g',
                        showarrow=False,
                        font=dict(size=18, color=COLORS['alert'], family=FONT,
                                   style='italic', weight='bold'))

    # Body axes (tilted) — both x and z
    # Body x (along float's forward direction, tilted by +pitch)
    bx = (1.6 * cos_p, 1.6 * sin_p)
    fig.add_annotation(x=bx[0], y=bx[1], ax=0, ay=0,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2,
                        arrowcolor=COLORS['accent'])
    fig.add_annotation(x=bx[0]+0.08, y=bx[1]+0.05, text='body  x',
                        showarrow=False,
                        font=dict(size=12, color=COLORS['accent'], family=FONT,
                                   weight='bold'))
    # Body z (perpendicular to body x, pointing up-left)
    bz = (-1.6 * sin_p, 1.6 * cos_p)
    fig.add_annotation(x=bz[0], y=bz[1], ax=0, ay=0,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=3, arrowsize=1.3, arrowwidth=2,
                        arrowcolor=COLORS['accent_2'])
    fig.add_annotation(x=bz[0]-0.12, y=bz[1]+0.1, text='body  z',
                        showarrow=False,
                        font=dict(size=12, color=COLORS['accent_2'], family=FONT,
                                   weight='bold'))

    # Gravity projections onto body x and body z (dotted lines from g-tip)
    # Project (0, -g_len) onto body x (direction = (cos_p, sin_p))
    proj_x_val = -g_len * sin_p  # = (0,-g_len) · (cos_p, sin_p) = -g_len * sin_p
    px_end = (proj_x_val * cos_p, proj_x_val * sin_p)
    fig.add_shape(type='line', x0=g_tip_x, y0=g_tip_y, x1=px_end[0], y1=px_end[1],
                   line=dict(color=COLORS['accent'], width=2, dash='dot'))
    # Project onto body z (direction = (-sin_p, cos_p))
    proj_z_val = -g_len * cos_p  # = (0,-g_len) · (-sin_p, cos_p) = -g_len * cos_p
    pz_end = (-proj_z_val * sin_p, proj_z_val * cos_p)
    fig.add_shape(type='line', x0=g_tip_x, y0=g_tip_y, x1=pz_end[0], y1=pz_end[1],
                   line=dict(color=COLORS['accent_2'], width=2, dash='dot'))

    # Labels for projections
    fig.add_annotation(x=px_end[0]*0.65, y=px_end[1]*0.65 - 0.18,
                        text='−g sin θ',
                        showarrow=False,
                        font=dict(size=12, color=COLORS['accent'], family=FONT,
                                   weight='bold'))
    fig.add_annotation(x=pz_end[0]*0.55 - 0.25, y=pz_end[1]*0.55,
                        text='−g cos θ',
                        showarrow=False,
                        font=dict(size=12, color=COLORS['accent_2'], family=FONT,
                                   weight='bold'))

    # Pitch angle arc between body-x and world-x (horizontal)
    arc_r = 0.6
    arc_angles = [pitch * i / 20 for i in range(21)]
    arc_x = [arc_r * math.cos(a) for a in arc_angles]
    arc_y = [arc_r * math.sin(a) for a in arc_angles]
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines',
                                line=dict(color=COLORS['text_dim'], width=1.5),
                                showlegend=False, hoverinfo='skip'))
    fig.add_annotation(x=(arc_r+0.18) * math.cos(pitch/2),
                        y=(arc_r+0.05) * math.sin(pitch/2),
                        text='θ',
                        showarrow=False,
                        font=dict(size=15, color=COLORS['text_dim'], family=FONT,
                                   style='italic', weight='bold'))

    fig.update_xaxes(range=[-2.0, 2.0], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-2.0, 1.8], visible=False, fixedrange=True,
                     scaleanchor='x', scaleratio=1)
    fig.update_layout(plot_layout(
        height=380,
        margin=dict(l=20, r=20, t=40, b=20),
        title=dict(text='When the float pitches by θ, world-vertical gravity '
                         'projects onto body-x and body-z',
                    font=dict(size=13)),
        showlegend=False,
    ))
    return fig


def make_adcp_geometry_figure(h_water=50.0, adcp_depth=10.0,
                                depth_bin_top=11.0, depth_bin_bottom=40.0,
                                n_bins=30):
    """Side view labelling local water depth, hull bottom (ADCP location),
    and the range of depth bins. Pedagogical schematic for the ADCP setup."""
    fig = go.Figure()

    # Water column
    fig.add_shape(type='rect', x0=-2.5, y0=-h_water, x1=2.5, y1=0,
                   fillcolor=COLORS['water'], opacity=0.22, line_width=0,
                   layer='below')
    # Water line
    fig.add_shape(type='line', x0=-2.5, y0=0, x1=2.5, y1=0,
                   line=dict(color=COLORS['accent'], width=1.6))
    fig.add_annotation(x=2.4, y=0, text='surface  z = 0',
                        showarrow=False, xanchor='right',
                        font=dict(size=11, color=COLORS['accent'], family=FONT),
                        yshift=10)
    # Seabed
    fig.add_shape(type='rect', x0=-2.5, y0=-h_water-2.5, x1=2.5, y1=-h_water,
                   fillcolor='#9C8B6E', opacity=0.85, line_width=0)
    fig.add_annotation(x=0, y=-h_water - 1.2, text='seabed',
                        showarrow=False,
                        font=dict(size=11, color='white', family=FONT,
                                   weight='bold'))
    # Local water depth label on the right side
    fig.add_annotation(x=-2.0, y=-h_water/2, ax=-2.0, ay=0,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.4,
                        arrowcolor=COLORS['text_dim'])
    fig.add_annotation(x=-2.0, y=-h_water/2, ax=-2.0, ay=-h_water,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.4,
                        arrowcolor=COLORS['text_dim'])
    fig.add_annotation(x=-2.25, y=-h_water/2,
                        text=f'local water depth<br>h = {h_water:.0f} m',
                        showarrow=False, xanchor='right',
                        font=dict(size=11, color=COLORS['text'], family=FONT))

    # Float TOP BODY — extends from above water down to hull bottom (= ADCP depth)
    # The whole top body is one continuous piece; ADCP sits at the bottom.
    fig.add_shape(type='rect',
                   x0=-0.55, y0=-adcp_depth, x1=0.55, y1=1.2,
                   fillcolor='#8E8A7B', line=dict(color=COLORS['text'], width=1.4))
    fig.add_annotation(x=0, y=(1.2 + (-adcp_depth))/2 + 1.0,
                        text='float<br>top body',
                        showarrow=False,
                        font=dict(size=10, color='white', family=FONT,
                                   weight='bold'))

    # ADCP transducer — small mark at the bottom of the top body
    fig.add_shape(type='rect',
                   x0=-0.45, y0=-adcp_depth-0.5, x1=0.45, y1=-adcp_depth,
                   fillcolor=COLORS['accent'], line=dict(color=COLORS['text'], width=1.4))
    fig.add_annotation(x=0, y=-adcp_depth-0.25, text='ADCP',
                        showarrow=False,
                        font=dict(size=10, color='white', family=FONT,
                                   weight='bold'))
    # Arrow + label pointing to the hull-bottom / ADCP line
    fig.add_annotation(
        x=0.55, y=-adcp_depth,
        ax=1.7, ay=-adcp_depth,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.2,
        arrowcolor=COLORS['accent'],
    )
    fig.add_annotation(x=1.75, y=-adcp_depth,
                        text=f'hull bottom  =  ADCP depth  ({adcp_depth:.0f} m below surface)',
                        showarrow=False, xanchor='left',
                        font=dict(size=11, color=COLORS['accent'],
                                   family=FONT, weight='bold'))

    # Neck → inertia mass (visual context, smaller below ADCP)
    fig.add_shape(type='rect',
                   x0=-0.12, y0=-adcp_depth-2.5,
                   x1=0.12, y1=-adcp_depth-0.5,
                   fillcolor='#6B6B6B', line=dict(color=COLORS['text'], width=1.0),
                   opacity=0.7)
    fig.add_annotation(x=0.25, y=-adcp_depth-1.5,
                        text='neck',
                        showarrow=False, xanchor='left',
                        font=dict(size=9, color=COLORS['text_dim'], family=FONT))
    # Inertia mass at the bottom of the neck (not the seabed)
    fig.add_shape(type='rect',
                   x0=-0.35, y0=-adcp_depth-3.3,
                   x1=0.35, y1=-adcp_depth-2.5,
                   fillcolor='#404040', line=dict(color=COLORS['text'], width=1.2))
    fig.add_annotation(x=0, y=-adcp_depth-2.9,
                        text='inertia',
                        showarrow=False,
                        font=dict(size=9, color='white', family=FONT,
                                   weight='bold'))

    # Depth-bin range bracket on the right
    fig.add_shape(type='line', x0=1.5, y0=-depth_bin_top,
                   x1=1.5, y1=-depth_bin_bottom,
                   line=dict(color=COLORS['accent_2'], width=2.5))
    fig.add_shape(type='line', x0=1.4, y0=-depth_bin_top,
                   x1=1.6, y1=-depth_bin_top,
                   line=dict(color=COLORS['accent_2'], width=2.5))
    fig.add_shape(type='line', x0=1.4, y0=-depth_bin_bottom,
                   x1=1.6, y1=-depth_bin_bottom,
                   line=dict(color=COLORS['accent_2'], width=2.5))
    fig.add_annotation(
        x=1.75,
        y=-(depth_bin_top + depth_bin_bottom)/2,
        text=f'depth bins<br>{int(depth_bin_top)} → {int(depth_bin_bottom)} m<br>({n_bins} bins)',
        showarrow=False, xanchor='left',
        font=dict(size=11, color=COLORS['accent_2'], family=FONT, weight='bold'))

    # Bin tick marks on the left
    for d_bin in [depth_bin_top, depth_bin_top+5, depth_bin_top+10,
                   depth_bin_top+15, depth_bin_top+20, depth_bin_bottom]:
        if d_bin > depth_bin_bottom:
            continue
        fig.add_shape(type='line', x0=-0.85, y0=-d_bin, x1=-0.55, y1=-d_bin,
                       line=dict(color=COLORS['accent_2'], width=0.8),
                       opacity=0.5)
        fig.add_annotation(x=-0.9, y=-d_bin, text=f'{int(d_bin)} m',
                            showarrow=False, xanchor='right',
                            font=dict(size=9, color=COLORS['text_dim'],
                                       family=FONT))

    fig.update_xaxes(range=[-3.0, 3.0], visible=False, fixedrange=True)
    fig.update_yaxes(range=[-h_water-3, 2.5], visible=False, fixedrange=True,
                     scaleanchor='x', scaleratio=0.10)
    fig.update_layout(plot_layout(
        height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text='ADCP geometry — vertical extent of the measurement column',
                    font=dict(size=13)),
        showlegend=False,
    ))
    return fig


def make_adcp_schematic():
    """Hull-mounted ADCP with downward-looking acoustic beams.
    Shows the sensing geometry: 4 angled beams covering a cone below the hull,
    measuring water velocity at depth bins."""
    fig = go.Figure()

    # Water (background)
    fig.add_shape(type='rect', x0=-3, y0=-7, x1=3, y1=0,
                   fillcolor=COLORS['water'], opacity=0.3, line_width=0,
                   layer='below')
    # Water line
    fig.add_shape(type='line', x0=-3, y0=0, x1=3, y1=0,
                   line=dict(color=COLORS['accent'], width=1.5))

    # Float hull (top body) sitting at water surface
    fig.add_shape(type='rect', x0=-1.2, y0=-0.3, x1=1.2, y1=0.6,
                   fillcolor='#8E8A7B', line=dict(color=COLORS['text'], width=1.5))
    fig.add_annotation(x=0, y=0.15, text='float hull',
                        showarrow=False, font=dict(size=10, color='white'))

    # ADCP sensor mounted at bottom of hull (small box, downward-facing)
    fig.add_shape(type='rect', x0=-0.25, y0=-0.45, x1=0.25, y1=-0.3,
                   fillcolor='#1F2937', line=dict(color='black', width=1.5))
    fig.add_annotation(x=0.4, y=-0.37, text=' ADCP', showarrow=False,
                        font=dict(size=10, color=COLORS['text']), xanchor='left')

    # 4 acoustic beams fanning out and down (Janus configuration)
    beam_color = COLORS['accent_2']
    beam_angles = [-25, -10, 10, 25]   # degrees from vertical
    import math
    for angle in beam_angles:
        ang_rad = math.radians(angle)
        x_end = 0.0 + 6.5 * math.sin(ang_rad)
        y_end = -0.45 - 6.5 * math.cos(ang_rad)
        fig.add_shape(type='line', x0=0.0, y0=-0.45, x1=x_end, y1=y_end,
                       line=dict(color=beam_color, width=1.2, dash='dot'))

    # Depth bins (horizontal lines) within the beam cone
    bin_depths = [-1.5, -2.5, -3.5, -4.5, -5.5]
    for d in bin_depths:
        # Width at this depth based on outer beam angle (25 deg)
        x_extent = abs((d + 0.45) * math.tan(math.radians(25)))
        fig.add_shape(type='line', x0=-x_extent, y0=d, x1=x_extent, y1=d,
                       line=dict(color=COLORS['text_dim'], width=0.6))
        # Current arrow at each bin (showing measured velocity)
        arrow_len = 0.6 if d > -3.5 else 0.3
        fig.add_annotation(
            x=x_extent + 0.3 + arrow_len, y=d, ax=x_extent + 0.3, ay=d,
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
            arrowcolor=COLORS['accent'], showarrow=True,
        )
        fig.add_annotation(x=-x_extent - 0.1, y=d, text=f'{abs(d):.0f}m',
                            showarrow=False, font=dict(size=9, color=COLORS['text_dim']),
                            xanchor='right')

    # Labels
    fig.add_annotation(x=0, y=-6.5,
                        text='each layer = one depth bin (water velocity measured here)',
                        showarrow=False, font=dict(size=10, color=COLORS['text_dim']),
                        xanchor='center')

    fig.update_layout(plot_layout(
        xaxis=dict(visible=False, range=[-3.5, 3.5]),
        yaxis=dict(visible=False, range=[-7.2, 1.2], scaleanchor='x', scaleratio=1),
        height=320, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    ))
    return fig


def make_pressure_schematic():
    """Float schematic with the 5 pressure sensor positions marked.

    Sensors get numbered badges placed *outside* the float body with
    leader lines pointing to their physical mounting location, so they
    don't visually pile up inside the small chamber drawings.
    """
    fig = go.Figure()

    # ---- Water (background) ----
    fig.add_shape(type='rect', x0=-3.5, y0=-8, x1=8, y1=0,
                   fillcolor=COLORS['water'], opacity=0.22, line_width=0,
                   layer='below')
    fig.add_shape(type='line', x0=-3.5, y0=0, x1=8, y1=0,
                   line=dict(color=COLORS['accent'], width=1.4))
    fig.add_annotation(x=-3.4, y=0.3,
                        text='atmosphere  (≈ 1.013 bar)',
                        showarrow=False, xanchor='left',
                        font=dict(size=10, color=COLORS['text_dim'], family=FONT))

    # ---- Depth ticks on the left ----
    for d_val, lbl in [(-1, '1 m'), (-3, '3 m'), (-6, '6 m')]:
        fig.add_shape(type='line', x0=-3.2, y0=d_val, x1=-2.6, y1=d_val,
                       line=dict(color=COLORS['text_dim'], width=0.8, dash='dot'))
        fig.add_annotation(x=-2.5, y=d_val, text=lbl,
                            showarrow=False, xanchor='left',
                            font=dict(size=9, color=COLORS['text_dim'],
                                       family=FONT))

    # ---- Float top body: stacked chambers, vertically generous ----
    body_x_half = 1.1
    # Dry-side cabin (top, above waterline)
    fig.add_shape(type='rect',
                   x0=-body_x_half, y0=1.0, x1=body_x_half, y1=2.6,
                   fillcolor='#F5F5DC',
                   line=dict(color=COLORS['text'], width=1.5))
    fig.add_annotation(x=0, y=2.05, text='dry-side cabin<br>(electronics)',
                        showarrow=False, align='center',
                        font=dict(size=10, color=COLORS['text'], family=FONT))

    # PTO chamber (middle, around waterline)
    fig.add_shape(type='rect',
                   x0=-body_x_half, y0=-1.6, x1=body_x_half, y1=1.0,
                   fillcolor='#FAE5E5',
                   line=dict(color=COLORS['text'], width=1.5))
    fig.add_annotation(x=0, y=-0.3, text='PTO chamber',
                        showarrow=False, align='center',
                        font=dict(size=11, color=COLORS['text'], family=FONT,
                                   weight='bold'))

    # Neck (long thin section below the body)
    fig.add_shape(type='rect',
                   x0=-0.20, y0=-6.5, x1=0.20, y1=-1.6,
                   fillcolor='#6B6B6B',
                   line=dict(color=COLORS['text'], width=1.2))

    # Inertia / anchor base at the bottom
    fig.add_shape(type='rect',
                   x0=-0.7, y0=-7.3, x1=0.7, y1=-6.5,
                   fillcolor='#404040',
                   line=dict(color=COLORS['text'], width=1.5))
    fig.add_annotation(x=0, y=-6.9, text='inertia mass',
                        showarrow=False,
                        font=dict(size=9, color='white', family=FONT))

    # ---- 5 sensor badges, placed OUTSIDE the float with leader lines ----
    # Each entry: (badge_x, badge_y, target_x, target_y, num, color)
    sensors = [
        ( 2.4, -1.0,  body_x_half, -1.0, '1', COLORS['accent_2']),  # ext hull 1m
        ( 2.4, -3.0,  0.2,         -3.0, '1', COLORS['accent_2']),  # ext hull 3m
        ( 2.4, -6.0,  0.2,         -6.0, '1', COLORS['accent_2']),  # ext hull 6m
        (-1.9,  0.3, -body_x_half,  0.3, '2', COLORS['alert']),     # PTO internal
        (-1.9, -1.0, -body_x_half, -1.0, '3', COLORS['warning']),   # differential
        (-1.9,  1.8, -body_x_half,  1.8, '4', '#7D3C98'),           # dry cabin
        ( 1.4, -7.0,  0.7,         -7.0, '5', '#7D3C98'),           # depth/draft
    ]
    for bx, by, tx, ty, num, color in sensors:
        # Leader line from badge to target
        fig.add_shape(type='line', x0=bx, y0=by, x1=tx, y1=ty,
                       line=dict(color=color, width=1.0))
        # Badge (numbered circle-equivalent annotation)
        fig.add_annotation(x=bx, y=by, text=f'<b>{num}</b>',
                            showarrow=False,
                            font=dict(size=11, color='white', family=FONT),
                            bgcolor=color, borderpad=5, borderwidth=0)

    # ---- Legend table on the right ----
    legend_items = [
        ('1', COLORS['accent_2'], 'External hull pressure × 3  (depths 1, 3, 6 m)'),
        ('2', COLORS['alert'],    'Internal PTO chamber  (working fluid)'),
        ('3', COLORS['warning'],  'Differential across turbine  (flow rate → power)'),
        ('4', '#7D3C98',          'Dry cabin pressure  (leak detection)'),
        ('5', '#7D3C98',          'Depth / draft sensor  (hull-bottom reference)'),
    ]
    for i, (num, color, lbl) in enumerate(legend_items):
        y_pos = 2.2 - i*1.0
        fig.add_annotation(x=3.5, y=y_pos, text=f'<b>{num}</b>',
                            showarrow=False,
                            font=dict(size=11, color='white', family=FONT),
                            bgcolor=color, borderpad=4, borderwidth=0)
        fig.add_annotation(x=3.85, y=y_pos, text=lbl,
                            showarrow=False, xanchor='left',
                            font=dict(size=10, color=COLORS['text'],
                                       family=FONT))

    fig.update_layout(plot_layout(
        xaxis=dict(visible=False, range=[-3.5, 8]),
        yaxis=dict(visible=False, range=[-8.0, 3.2],
                    scaleanchor='x', scaleratio=1),
        height=460, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    ))
    return fig


def make_pto_geometry_figure():
    """Schematic showing the difference between **capture width** (external,
    horizontal extent of float at the waterline) and **piston area /
    stroke length** (internal hydraulic cylinder geometry).

    Uses the same float shape as the Overview tab: top body + neck +
    inertia mass.
    """
    fig = go.Figure()

    # ---------- Background: water + waterline ----------
    fig.add_shape(type='rect',
                   x0=-4.5, y0=-5.5, x1=4.5, y1=0,
                   fillcolor=COLORS['water'], opacity=0.18, line_width=0,
                   layer='below')
    fig.add_shape(type='line', x0=-4.5, y0=0, x1=4.5, y1=0,
                   line=dict(color=COLORS['accent'], width=1.4))
    fig.add_annotation(x=-4.4, y=0, text='waterline', showarrow=False,
                        xanchor='left', yshift=10,
                        font=dict(size=10, color=COLORS['accent'],
                                   family=FONT))

    # ---------- Float top body (Overview shape: rounded-top rectangle) ----------
    # Above-water dome
    top_w = 1.4    # half-width at waterline
    top_h = 1.1    # height above waterline
    fig.add_shape(type='path',
                   path=f"M {-top_w},0 L {-top_w},{top_h-0.4} "
                        f"Q {-top_w},{top_h} {-top_w+0.4},{top_h} "
                        f"L {top_w-0.4},{top_h} "
                        f"Q {top_w},{top_h} {top_w},{top_h-0.4} "
                        f"L {top_w},0 Z",
                   fillcolor='#8E8A7B',
                   line=dict(color=COLORS['text'], width=1.4))
    # Submerged body (rectangle below waterline)
    body_bot = -1.4
    fig.add_shape(type='rect',
                   x0=-top_w, y0=body_bot, x1=top_w, y1=0,
                   fillcolor='#8E8A7B',
                   line=dict(color=COLORS['text'], width=1.4))
    fig.add_annotation(x=0, y=0.55, text='float<br>top body',
                        showarrow=False,
                        font=dict(size=10, color='white', family=FONT,
                                   weight='bold'))

    # ---------- Neck (thin vertical rod) ----------
    neck_top = body_bot
    neck_bot = -3.6
    fig.add_shape(type='rect',
                   x0=-0.10, y0=neck_bot, x1=0.10, y1=neck_top,
                   fillcolor='#6B6B6B', line=dict(color=COLORS['text'], width=1.0))
    fig.add_annotation(x=0.18, y=(neck_top + neck_bot)/2,
                        text='neck',
                        showarrow=False, xanchor='left',
                        font=dict(size=9, color=COLORS['text_dim'],
                                   family=FONT))

    # ---------- Inertia mass at the bottom ----------
    fig.add_shape(type='rect',
                   x0=-0.55, y0=-4.5, x1=0.55, y1=neck_bot,
                   fillcolor='#404040',
                   line=dict(color=COLORS['text'], width=1.2))
    fig.add_annotation(x=0, y=(-4.5 + neck_bot)/2,
                        text='inertia',
                        showarrow=False,
                        font=dict(size=9, color='white', family=FONT,
                                   weight='bold'))

    # ---------- INTERNAL: piston cylinder inside the top body ----------
    # Cylinder bounds (inside the submerged part of the top body)
    cyl_x = 0.38
    cyl_top = -0.15
    cyl_bot = -1.30
    fig.add_shape(type='rect',
                   x0=-cyl_x, y0=cyl_bot, x1=cyl_x, y1=cyl_top,
                   fillcolor='#F8F0E0',
                   line=dict(color=COLORS['text'], width=1.2))
    # The piston itself (horizontal bar inside cylinder)
    piston_y = -0.65
    fig.add_shape(type='rect',
                   x0=-cyl_x+0.04, y0=piston_y-0.06,
                   x1=cyl_x-0.04,  y1=piston_y+0.06,
                   fillcolor=COLORS['accent_2'],
                   line=dict(color=COLORS['text'], width=1.0))
    # Piston rod going up out of the cylinder
    fig.add_shape(type='line',
                   x0=0, y0=piston_y+0.06, x1=0, y1=cyl_top+0.3,
                   line=dict(color=COLORS['text'], width=1.4))

    # ---------- LABEL 1: capture width (external, at the waterline) ----------
    # Horizontal double-arrow along the waterline
    fig.add_annotation(
        x=top_w, y=0.12, ax=-top_w, ay=0.12,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=1.3,
        arrowcolor=COLORS['accent'],
    )
    fig.add_annotation(
        x=-top_w, y=0.12, ax=top_w, ay=0.12,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowsize=1.0, arrowwidth=1.3,
        arrowcolor=COLORS['accent'],
    )
    fig.add_annotation(x=2.5, y=0.12,
                        text='<b>capture width  w_c</b>  '
                             '<i>(external — wave field intercepted)</i>',
                        showarrow=False, xanchor='left',
                        font=dict(size=11, color=COLORS['accent'],
                                   family=FONT))

    # ---------- LABEL 2: piston area (internal) ----------
    fig.add_annotation(
        x=cyl_x+0.04, y=piston_y, ax=2.0, ay=piston_y,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.1,
        arrowcolor=COLORS['accent_2'],
    )
    fig.add_annotation(x=2.1, y=piston_y,
                        text='<b>piston area  A_piston</b>  '
                             '<i>(internal — cylinder cross-section)</i>',
                        showarrow=False, xanchor='left',
                        font=dict(size=11, color=COLORS['accent_2'],
                                   family=FONT))

    # ---------- LABEL 3: stroke length (internal, vertical extent) ----------
    fig.add_annotation(
        x=-cyl_x-0.08, y=cyl_top-0.05, ax=-cyl_x-0.08, ay=cyl_bot+0.05,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.2,
        arrowcolor=COLORS['warning'],
    )
    fig.add_annotation(
        x=-cyl_x-0.08, y=cyl_bot+0.05, ax=-cyl_x-0.08, ay=cyl_top-0.05,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowsize=1, arrowwidth=1.2,
        arrowcolor=COLORS['warning'],
    )
    fig.add_annotation(x=-2.5, y=(cyl_top+cyl_bot)/2,
                        text='<b>stroke L</b>  '
                             '<i>(internal — piston travel)</i>',
                        showarrow=False, xanchor='right',
                        font=dict(size=11, color=COLORS['warning'],
                                   family=FONT))

    # ---------- Incoming wave label (top-left) ----------
    fig.add_annotation(x=-4.4, y=0.65,
                        text='incoming<br>wave energy  →',
                        showarrow=False, xanchor='left',
                        font=dict(size=10, color=COLORS['text_dim'],
                                   family=FONT))

    fig.update_layout(plot_layout(
        xaxis=dict(visible=False, range=[-4.5, 4.5]),
        yaxis=dict(visible=False, range=[-5.0, 1.5],
                    scaleanchor='x', scaleratio=1),
        height=420, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    ))
    return fig


def make_sdof_diagram():
    """Schematic of the SDOF abstraction: mass-spring-damper."""
    fig = go.Figure()
    # Wall (top, fixed reference)
    fig.add_shape(type='rect', x0=-1.5, y0=2.7, x1=1.5, y1=3.0,
                   fillcolor='#333', line_width=0)
    # Diagonal hatching marks for the fixed wall
    for i in range(7):
        xs = -1.4 + i*0.5
        fig.add_shape(type='line', x0=xs, y0=3.0, x1=xs+0.18, y1=3.18,
                       line=dict(color='#333', width=1))
    # Spring (zigzag)
    fig.add_shape(type='line', x0=-0.5, y0=2.7, x1=-0.5, y1=2.5,
                   line=dict(color=COLORS['text'], width=2))
    zig = [(-0.5, 2.5), (-0.7, 2.3), (-0.3, 2.1), (-0.7, 1.9),
           (-0.3, 1.7), (-0.7, 1.5), (-0.3, 1.3), (-0.5, 1.1)]
    for (x0, y0), (x1, y1) in zip(zig[:-1], zig[1:]):
        fig.add_shape(type='line', x0=x0, y0=y0, x1=x1, y1=y1,
                       line=dict(color=COLORS['text'], width=2))
    fig.add_shape(type='line', x0=-0.5, y0=1.1, x1=-0.5, y1=0.8,
                   line=dict(color=COLORS['text'], width=2))
    fig.add_annotation(x=-1.0, y=2.0, text='spring k', showarrow=False,
                        font=dict(size=11, color=COLORS['text']))

    # Dashpot
    fig.add_shape(type='line', x0=0.5, y0=2.7, x1=0.5, y1=2.0,
                   line=dict(color=COLORS['text'], width=2))
    fig.add_shape(type='rect', x0=0.3, y0=1.5, x1=0.7, y1=2.0,
                   fillcolor='white', line=dict(color=COLORS['text'], width=2))
    fig.add_shape(type='line', x0=0.5, y0=1.7, x1=0.5, y1=0.8,
                   line=dict(color=COLORS['text'], width=2))
    fig.add_shape(type='line', x0=0.2, y0=1.7, x1=0.8, y1=1.7,
                   line=dict(color=COLORS['text'], width=2))
    fig.add_annotation(x=1.0, y=1.85, text='damper c', showarrow=False,
                        font=dict(size=11, color=COLORS['text']))

    # Mass block
    fig.add_shape(type='rect', x0=-1.0, y0=0.0, x1=1.0, y1=0.8,
                   fillcolor=COLORS['accent'], line=dict(color=COLORS['text'], width=1.5))
    fig.add_annotation(x=0, y=0.4, text='mass m', showarrow=False,
                        font=dict(size=13, color='white', family=FONT))

    # x(t) coordinate arrow
    fig.add_annotation(x=1.4, y=0.6, ax=1.4, ay=0.2,
                        arrowhead=3, arrowsize=1.5, arrowwidth=2,
                        arrowcolor=COLORS['text'], showarrow=True)
    fig.add_annotation(x=1.7, y=0.4, text='x(t)', showarrow=False,
                        font=dict(size=13, color=COLORS['text'], family=FONT))

    # Forcing arrow
    fig.add_annotation(x=0, y=-0.3, ax=0, ay=-0.9,
                        arrowhead=3, arrowsize=1.5, arrowwidth=2.5,
                        arrowcolor=COLORS['alert'], showarrow=True)
    fig.add_annotation(x=0.5, y=-0.6, text='F(t)', showarrow=False,
                        font=dict(size=13, color=COLORS['alert'], family=FONT))

    fig.update_layout(plot_layout(
        xaxis=dict(visible=False, range=[-2.5, 2.5]),
        yaxis=dict(visible=False, range=[-1.2, 3.4],
                    scaleanchor='x', scaleratio=1),
        height=320, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    ))
    return fig


def make_float_sdof_figure():
    """SDOF abstraction *mapped onto the actual float shape*.

    Same equation of motion (m ẍ + c ẋ + k x = F), but drawn on the real
    float (top body + neck + inertia — same shape as the Overview tab).
    Each symbol is labelled on the diagram with a leader line; the
    meanings live in a compact legend box in the bottom-right corner,
    so the diagram itself stays uncluttered.
    """
    fig = go.Figure()

    # ---- Background: water + waterline ----
    fig.add_shape(type='rect',
                   x0=-5.0, y0=-5.5, x1=5.0, y1=0,
                   fillcolor=COLORS['water'], opacity=0.18, line_width=0,
                   layer='below')
    fig.add_shape(type='line', x0=-5.0, y0=0, x1=5.0, y1=0,
                   line=dict(color=COLORS['accent'], width=1.4))
    fig.add_annotation(x=-4.9, y=0, text='waterline',
                        showarrow=False, xanchor='left', yshift=10,
                        font=dict(size=10, color=COLORS['accent'],
                                   family=FONT))

    # ---- Float top body (rounded-top rectangle, same as Overview) ----
    top_w = 1.4
    top_h = 1.1
    fig.add_shape(type='path',
                   path=f"M {-top_w},0 L {-top_w},{top_h-0.4} "
                        f"Q {-top_w},{top_h} {-top_w+0.4},{top_h} "
                        f"L {top_w-0.4},{top_h} "
                        f"Q {top_w},{top_h} {top_w},{top_h-0.4} "
                        f"L {top_w},0 Z",
                   fillcolor='#8E8A7B',
                   line=dict(color=COLORS['text'], width=1.4))
    body_bot = -1.4
    fig.add_shape(type='rect',
                   x0=-top_w, y0=body_bot, x1=top_w, y1=0,
                   fillcolor='#8E8A7B',
                   line=dict(color=COLORS['text'], width=1.4))

    # ---- Neck ----
    neck_top = body_bot
    neck_bot = -3.6
    fig.add_shape(type='rect',
                   x0=-0.10, y0=neck_bot, x1=0.10, y1=neck_top,
                   fillcolor='#6B6B6B',
                   line=dict(color=COLORS['text'], width=1.0))

    # ---- Inertia mass at the bottom ----
    fig.add_shape(type='rect',
                   x0=-0.55, y0=-4.5, x1=0.55, y1=neck_bot,
                   fillcolor='#404040',
                   line=dict(color=COLORS['text'], width=1.2))

    # ---- Symbol labels ON the diagram (short, just the symbol) ----
    # m on the top body
    fig.add_annotation(x=0, y=0.5,
                        text='<b style="font-size:18px; color:#FFF">m</b>',
                        showarrow=False,
                        font=dict(family=FONT))

    # k at the waterline (hydrostatic) — short label with leader
    fig.add_annotation(
        x=top_w+0.05, y=-0.15, ax=top_w+0.9, ay=-0.15,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.0,
        arrowcolor=COLORS['accent'],
    )
    fig.add_annotation(x=top_w+0.95, y=-0.15,
                        text='<b style="font-size:14px">k</b>',
                        showarrow=False, xanchor='left',
                        font=dict(color=COLORS['accent'], family=FONT))

    # c on the left side of the body (damping)
    fig.add_annotation(
        x=-top_w-0.05, y=-0.8, ax=-top_w-0.85, ay=-0.8,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.0,
        arrowcolor=COLORS['warning'],
    )
    fig.add_annotation(x=-top_w-0.9, y=-0.8,
                        text='<b style="font-size:14px">c</b>',
                        showarrow=False, xanchor='right',
                        font=dict(color=COLORS['warning'], family=FONT))

    # F(t) — wave forcing from the left
    fig.add_annotation(
        x=-top_w-0.05, y=0.3, ax=-3.0, ay=0.3,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowsize=1.4, arrowwidth=2.0,
        arrowcolor=COLORS['alert'],
    )
    fig.add_annotation(x=-3.05, y=0.5,
                        text='<b style="font-size:14px">F(t)</b>',
                        showarrow=False, xanchor='right',
                        font=dict(color=COLORS['alert'], family=FONT))

    # x(t) — heave on top of the float
    fig.add_annotation(
        x=0, y=top_h+0.45, ax=0, ay=top_h-0.35,
        xref='x', yref='y', axref='x', ayref='y',
        showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.8,
        arrowcolor=COLORS['text'],
    )
    fig.add_annotation(x=0.22, y=top_h+0.25,
                        text='<b style="font-size:13px">x(t)</b>',
                        showarrow=False, xanchor='left',
                        font=dict(color=COLORS['text'], family=FONT))

    fig.update_layout(plot_layout(
        xaxis=dict(visible=False, range=[-3.0, 3.0]),
        yaxis=dict(visible=False, range=[-5.0, 1.9],
                    scaleanchor='x', scaleratio=1),
        height=380, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    ))
    return fig


# ========================================================================
# PRE-COMPUTED DATA
# Generate all sensor data once at module load. Avoids re-computing in
# callbacks (important for Render free-tier performance).
# ========================================================================
np.random.seed(42)


def make_fatigue_data(
    sigma_limit_air=80.0, env_factor=0.75,
    yield_str=250.0, UTS=450.0,
):
    """
    V2 framework: 6 sea-state scenarios, each fully analysed for fatigue +
    hard limits. Pre-computes everything needed by the Fatigue tab,
    including BS 7608 standard reference table and per-scenario
    physical-sea-state metadata.

    Parameters (all tunable via UI sliders):
      sigma_limit_air  — in-air fatigue limit at 10⁷ cycles (MPa). BS 7608
                          Class B = 100, D = 80 (default), F = 56, W = 25.
      env_factor       — environment reduction multiplier on σ_limit.
                          Air = 1.00, seawater + cathodic protection = 0.75
                          (default), seawater + free corrosion = 0.60.
      yield_str        — yield strength of structural steel (MPa). S235 = 250 (default).
      UTS              — ultimate tensile strength (MPa). S235 = 450 (default).
    """
    # ============== Material ==============
    E = 200_000.0   # MPa  (= 200 GPa)

    # ============== BS 7608 standard reference (full table for sidebar/expandable) ==============
    bs7608_classes = {
        'B': dict(sigma_limit_air=100.0, N_limit=1e7, m=3.0,
                   description='Plain plate, machined / ground edge — best class'),
        'D': dict(sigma_limit_air=80.0,  N_limit=1e7, m=3.0,
                   description='Transverse butt weld, full penetration, ground flush'),
        'F': dict(sigma_limit_air=56.0,  N_limit=1e7, m=3.0,
                   description='Fillet / load-carrying weld — typical offshore'),
        'W': dict(sigma_limit_air=25.0,  N_limit=1e7, m=3.0,
                   description='Cruciform load-carrying fillet — worst class'),
    }
    env_factors = {
        'air':                     1.00,
        'seawater_cathodic':       0.75,   # typical, with cathodic protection
        'seawater_free_corrosion': 0.60,   # unprotected
    }

    # Configuration: σ_limit = σ_limit_air × env_factor (defaults 80 × 0.75 = 60 MPa)
    weld_class_used = 'D'
    environment     = 'seawater_cathodic'
    sigma_limit     = sigma_limit_air * env_factor
    N_limit         = 1e7
    m_sn            = 3.0

    # ============== Hard limits (typical structural steel S235) ==============
    yield_sf = 1.5
    UTS_sf = 2.5

    # ============== Sampling / filter ==============
    fs = 10
    duration = 30 * 60
    nyquist = fs / 2
    cutoff = 0.5
    b_filt, a_filt = butter(4, cutoff/nyquist, btype='low')

    # ============== Signal generator ==============
    def make_strain(amp, env_d, noise, seed=42):
        rng = np.random.default_rng(seed)
        t = np.arange(0, duration, 1/fs)
        carrier = amp * np.sin(2*np.pi*t/8.0)
        env = 1 + env_d * np.sin(2*np.pi*t/300)
        return t, carrier*env + 200 + rng.normal(0, noise, len(t))

    # ============== S-N curve helpers ==============
    def cycles_to_failure(sigma):
        if sigma <= sigma_limit:
            return np.inf
        return N_limit * (sigma_limit / sigma) ** m_sn

    def damage_per_cycle(sigma):
        if sigma <= sigma_limit:
            return 0.0
        return (sigma / sigma_limit) ** m_sn / N_limit

    # ============== Scenario definitions WITH PHYSICAL MAPPING ==============
    # Each scenario maps to a real-world sea state. Strain amplitudes are
    # the result of wave height + wave loading on the float's neck weld.
    scenarios = {
        'Calm sea': dict(
            strain_amplitude=30,  envelope_depth=0.2, noise_level=10,
            Hs_m=0.5, Tp_s=6.0, beaufort='0-2',
            description='Glassy / very light waves, minimal device motion',
        ),
        'Light operation': dict(
            strain_amplitude=60,  envelope_depth=0.3, noise_level=12,
            Hs_m=1.2, Tp_s=7.0, beaufort='3',
            description='Light breeze, small waves, comfortable operating sea',
        ),
        'Moderate': dict(
            strain_amplitude=120, envelope_depth=0.4, noise_level=15,
            Hs_m=2.5, Tp_s=8.0, beaufort='4-5',
            description='Typical Pacific working day, 15-20 kt winds',
        ),
        'Heavy seas': dict(
            strain_amplitude=250, envelope_depth=0.5, noise_level=20,
            Hs_m=4.5, Tp_s=9.0, beaufort='6-7',
            description='Strong winds, large waves, near operational limit',
        ),
        'Storm': dict(
            strain_amplitude=400, envelope_depth=0.6, noise_level=30,
            Hs_m=7.0, Tp_s=11.0, beaufort='8+',
            description='Gale, survival mode — device should still hold',
        ),
        'Sensor noisy': dict(
            strain_amplitude=120, envelope_depth=0.4, noise_level=80,
            Hs_m=2.5, Tp_s=8.0, beaufort='4-5',
            description='SAME waves as Moderate, but degraded strain gauge '
                         '(5× noise) — shows Method 3 is noise-robust',
        ),
    }

    results = {}
    for name, p in scenarios.items():
        t, strain_us = make_strain(p['strain_amplitude'], p['envelope_depth'],
                                     p['noise_level'])
        stress_raw = strain_us * E / 1e6
        stress_filt = filtfilt(b_filt, a_filt, stress_raw)

        cycles_raw = list(rainflow.count_cycles(stress_raw))
        cycles_filt = list(rainflow.count_cycles(stress_filt))

        n_raw = sum(c for _, c in cycles_raw)
        n_filt = sum(c for _, c in cycles_filt)

        n_damaging = sum(c for s, c in cycles_filt if s > sigma_limit)
        n_safe = n_filt - n_damaging

        d_30min = sum(damage_per_cycle(s) * c for s, c in cycles_filt)
        d_per_yr = d_30min * 2 * 24 * 365
        d_20yr = d_per_yr * 20
        years_to_fail = (1/d_per_yr) if d_per_yr > 0 else float('inf')

        max_stress = float(np.max(np.abs(stress_raw - np.mean(stress_raw))))
        max_cycle_range = max((s for s, _ in cycles_raw), default=0)

        # ====== Monte Carlo: peak stress distribution ======
        # 200 independent 30-min realizations, different RNG seed each.
        # Captures variability of σ_peak across "different days at sea".
        N_mc = 200
        mc_peaks = np.empty(N_mc)
        for k in range(N_mc):
            _, strain_us_k = make_strain(p['strain_amplitude'],
                                          p['envelope_depth'],
                                          p['noise_level'],
                                          seed=1000 + k)
            stress_raw_k = strain_us_k * E / 1e6
            mc_peaks[k] = float(np.max(np.abs(stress_raw_k
                                                - np.mean(stress_raw_k))))
        mc_peak_mean = float(np.mean(mc_peaks))
        mc_peak_p95  = float(np.percentile(mc_peaks, 95))
        mc_peak_max  = float(np.max(mc_peaks))

        exceeds_yield_sf = mc_peak_max > yield_str / yield_sf
        exceeds_UTS_sf = mc_peak_max > UTS / UTS_sf
        margin_y = (yield_str / yield_sf) / mc_peak_max if mc_peak_max > 0 else float('inf')
        margin_uts = (UTS / UTS_sf) / mc_peak_max if mc_peak_max > 0 else float('inf')

        fatigue_pass = d_20yr < 1.0
        yield_pass = not exceeds_yield_sf
        uts_pass = not exceeds_UTS_sf
        overall_pass = fatigue_pass and yield_pass and uts_pass

        results[name] = dict(
            params=p, t=t,
            strain_us=strain_us, stress_raw=stress_raw, stress_filt=stress_filt,
            cycles_raw=cycles_raw, cycles_filt=cycles_filt,
            n_raw=n_raw, n_filt=n_filt, n_damaging=n_damaging, n_safe=n_safe,
            d_30min=d_30min, d_per_yr=d_per_yr, d_20yr=d_20yr,
            years_to_fail=years_to_fail,
            max_stress=max_stress, max_cycle_range=max_cycle_range,
            margin_y=margin_y, margin_uts=margin_uts,
            fatigue_pass=fatigue_pass, yield_pass=yield_pass, uts_pass=uts_pass,
            overall_pass=overall_pass,
            mc_peaks=mc_peaks, mc_peak_mean=mc_peak_mean,
            mc_peak_p95=mc_peak_p95, mc_peak_max=mc_peak_max,
            mc_n_runs=N_mc,
        )

    return dict(
        scenarios=results,
        bs7608_classes=bs7608_classes,
        env_factors=env_factors,
        weld_class_used=weld_class_used,
        environment=environment,
        material=dict(
            E=E, sigma_limit_air=sigma_limit_air, env_factor=env_factor,
            sigma_limit=sigma_limit, N_limit=N_limit, m_sn=m_sn,
            yield_str=yield_str, UTS=UTS, yield_sf=yield_sf, UTS_sf=UTS_sf,
        ),
        filter_params=dict(fs=fs, cutoff=cutoff, order=4),
        cycles_to_failure_fn=cycles_to_failure,
        damage_fn=damage_per_cycle,
    )


def make_strain_data(m=200_000.0, k=50_000.0, zeta=0.10, F_demo_N=5000.0):
    """V4 resonance demo: 3 scenarios at same forcing, different periods.
    Also pre-computes white-noise forcing + response for the 4-panel FFT
    figure (forcing time + response time + spectra + anomaly).

    Parameters (all tunable via UI sliders):
      m         — effective mass (kg)
      k         — restoring stiffness (N/m)
      zeta      — damping ratio (dimensionless)
      F_demo_N  — peak forcing amplitude used by the resonance demo
                  scenarios + the |H| table examples (N)
    """
    # SDOF parameters — c derived from ζ to keep things self-consistent
    c = 2 * zeta * np.sqrt(m * k)
    omega_n = np.sqrt(k / m)
    T_n = 2 * np.pi / omega_n
    strain_per_meter = 1000.0

    fs = 10
    duration = 600   # 10 minutes per scenario
    t = np.arange(0, duration, 1/fs)

    def transfer_function(omega):
        return 1.0 / np.sqrt((k - m * omega**2)**2 + (c * omega)**2)

    def apply_dynamics(F_t, m_use=m):
        """Apply SDOF dynamics to a forcing time series via FFT-based ODE solver."""
        N = len(F_t)
        F_fft_arr = fft(F_t)
        omegas = 2 * np.pi * fftfreq(N, 1/fs)
        denom = k - m_use * omegas**2 + 1j * c * omegas
        return np.real(np.fft.ifft(F_fft_arr / denom))

    F_amp = 5000.0
    scenarios = []
    for T_wave, label, color in [
        (4.0, 'A: T=4s (off-resonance, fast)', COLORS['plot_3']),
        (T_n, f'B: T={T_n:.1f}s (RESONANCE)', COLORS['alert']),
        (30.0, 'C: T=30s (quasi-static, slow)', COLORS['accent_2']),
    ]:
        carrier = F_amp * np.sin(2*np.pi*t/T_wave)
        env = 1 + 0.4 * np.sin(2*np.pi*t/300)
        F_t = carrier * env
        x_t = apply_dynamics(F_t)
        strain_clean = x_t * strain_per_meter
        strain_noisy = strain_clean + np.random.normal(0, 15, len(t))
        scenarios.append({
            't': t, 'F_t': F_t, 'strain_clean': strain_clean,
            'strain_noisy': strain_noisy, 'T_wave': T_wave,
            'label': label, 'color': color,
            'max_stress': float(np.max(np.abs(strain_clean)) * 0.2),
            'H_val': transfer_function(2*np.pi/T_wave) * 1e6,  # μm/N
        })

    # Transfer function curve
    omega_range = np.linspace(0.05, 3, 500)
    H_curve = transfer_function(omega_range) * 1e6

    # === 4-panel FFT identification: white-noise forcing ===
    t_long = np.arange(0, 3600, 1/fs)   # 60 min for good FFT resolution
    np.random.seed(123)
    F_white = np.random.normal(0, F_amp, len(t_long))   # white-noise forcing
    x_white = apply_dynamics(F_white)
    strain_white = x_white * strain_per_meter + np.random.normal(0, 15, len(t_long))

    # FFTs of both forcing and response
    N_long = len(t_long)
    freqs_hz = fftfreq(N_long, 1/fs)[:N_long//2]
    periods_fft = 1 / np.where(freqs_hz > 0, freqs_hz, np.inf)
    F_fft = np.abs(fft(F_white))[:N_long//2]
    S_fft = np.abs(fft(strain_white))[:N_long//2]

    def smooth(x, k_window=21):
        return np.convolve(x, np.ones(k_window)/k_window, mode='same')

    F_fft_smooth = smooth(F_fft, 21)
    S_smooth = smooth(S_fft, 21)
    mask_search = (periods_fft > 5) & (periods_fft < 25)
    peak_idx = np.argmax(S_smooth[mask_search])
    peak_period = periods_fft[mask_search][peak_idx]

    # === Anomaly scenario: 10% mass increase from water ingress ===
    m_fault = m * 1.10
    omega_n_fault = np.sqrt(k / m_fault)
    T_n_fault = 2 * np.pi / omega_n_fault
    x_fault = apply_dynamics(F_white, m_use=m_fault)
    strain_fault = x_fault * strain_per_meter + np.random.normal(0, 15, len(t_long))
    S_fft_fault = np.abs(fft(strain_fault))[:N_long//2]
    S_fault_smooth = smooth(S_fft_fault, 21)
    peak_fault_idx = np.argmax(S_fault_smooth[mask_search])
    peak_period_fault = periods_fft[mask_search][peak_fault_idx]

    return {
        'scenarios': scenarios,
        'omega_n': omega_n, 'T_n': T_n,
        'omega_range': omega_range, 'H_curve': H_curve,
        # 4-panel FFT data
        't_long': t_long, 'F_white': F_white, 'strain_white': strain_white,
        'periods_fft': periods_fft, 'F_fft_smooth': F_fft_smooth,
        'S_smooth': S_smooth, 'peak_period': peak_period,
        'S_fault_smooth': S_fault_smooth, 'peak_period_fault': peak_period_fault,
        'T_n_fault': T_n_fault, 'm_fault': m_fault,
        'params': {'m': m, 'k': k, 'c': c, 'zeta': c/(2*np.sqrt(m*k)),
                    'F_amp': F_amp, 'strain_per_meter': strain_per_meter,
                    'fs': fs,
                    # Single source of truth for the demo forcing
                    # used in the |H| table examples + resonance scenarios.
                    'F_demo_N': F_demo_N,
                    },
    }


def make_motion_data(
    A_wave=2.0, T_wave=8.0, wave_direction_deg=30.0,
    RAO_heave=0.60, RAO_surge=0.55, RAO_sway=0.50,
    phase_heave=0.0, phase_surge=0.3, phase_sway=1.5,
    roll_amp_deg=5.0, pitch_amp_deg_imu=3.0,
    roll_phase=0.0, pitch_phase=0.5,
    yaw_drift_rate=0.05,
):
    """
    Accelerometer (3-axis) + gyro + IMU fusion data for the Motion tab.

    Parameters (all tunable via UI sliders):
      ----- Wave field -----
      A_wave              — wave amplitude (m)
      T_wave              — wave period (s)
      wave_direction_deg  — incidence angle θ from body x-axis (deg)
      ----- Float translation response -----
      RAO_heave/surge/sway      — response-amplitude operators (dimensionless)
      phase_heave/surge/sway    — phase offsets of each translation channel (rad)
      ----- Float rotation response -----
      roll_amp_deg, pitch_amp_deg_imu  — angular amplitudes (deg)
      roll_phase, pitch_phase          — rotation phase offsets (rad)
      yaw_drift_rate                   — gyro yaw drift (deg/s)
    """
    fs = 100                                  # accelerometer sample rate, Hz
    duration = 60
    t = np.arange(0, duration, 1/fs)
    n = len(t)

    # ===== Wave parameters =====
    omega   = 2*np.pi/T_wave
    g_const = 9.81

    # ===== Wave-direction angle =====
    theta = np.deg2rad(wave_direction_deg)

    # ===== Sensor noise =====
    noise_g = 0.008

    # =================================================================
    # Block 1: 3-axis acceleration
    # =================================================================
    az_amp_mps2 = RAO_heave * A_wave * omega**2
    az_amp_g    = az_amp_mps2 / g_const

    ax_amp_mps2 = RAO_surge * A_wave * omega**2 * np.cos(theta)
    ax_amp_g    = ax_amp_mps2 / g_const
    ay_amp_mps2 = RAO_sway  * A_wave * omega**2 * np.sin(theta)
    ay_amp_g    = ay_amp_mps2 / g_const

    az_motion_mps2 = -RAO_heave * A_wave * omega**2 * np.sin(omega*t + phase_heave)
    az_reading_g   = 1.0 + az_motion_mps2 / g_const + np.random.normal(0, noise_g, n)

    ax_reading_g = ax_amp_g * np.cos(omega*t + phase_surge) \
                    + np.random.normal(0, noise_g, n)
    ay_reading_g = ay_amp_g * np.cos(omega*t + phase_sway) \
                    + np.random.normal(0, noise_g, n)

    # =================================================================
    # Block 2: 4 anomaly patterns on a_z
    # =================================================================
    # Normal: use the actual az_reading from Block 1
    signal_normal = az_reading_g.copy()

    # Anomaly 1: Bias drift (hardware aging / temperature / calibration loss)
    drift_rate_g_per_s = 0.003
    signal_drift = az_reading_g + drift_rate_g_per_s * t
    drift_trend_line = 1.0 + drift_rate_g_per_s * t

    # Anomaly 2: Saturation — storm waves exceed sensor range
    storm_mult = 5.0
    storm_az_motion = -RAO_heave * (A_wave * storm_mult) * omega**2 * np.sin(omega*t)
    storm_az_reading = 1.0 + storm_az_motion / g_const
    sat_max = 1.18
    sat_min = 0.82
    signal_saturated = np.clip(storm_az_reading, sat_min, sat_max) \
                        + np.random.normal(0, 0.005, n)

    # Anomaly 3: Stuck — electronics dead, constant value
    stuck_value = 0.97
    signal_stuck = np.ones(n) * stuck_value + np.random.normal(0, 0.003, n)

    # =================================================================
    # Block 3: IMU + sensor fusion
    # Strategy: ground-truth roll/pitch/yaw → derive gyro + accel readings
    # → recover orientation via imufusion (or complementary filter fallback)
    # Note: roll_amp_deg, pitch_amp_deg_imu, roll_phase, pitch_phase,
    # yaw_drift_rate are now kwargs at the top of make_motion_data().
    # =================================================================
    gyro_noise_dps    = 0.2                   # deg/s RMS
    accel_noise_imu_g = 0.01                  # g RMS

    # Ground truth angles (degrees)
    roll_true_deg  = roll_amp_deg     * np.sin(omega*t + roll_phase)
    pitch_true_deg = pitch_amp_deg_imu * np.cos(omega*t + pitch_phase)
    yaw_true_deg   = yaw_drift_rate * t

    roll_rad  = np.deg2rad(roll_true_deg)
    pitch_rad = np.deg2rad(pitch_true_deg)

    # Gyro readings (derivative of angles + noise)
    gyro_x = roll_amp_deg * omega * np.cos(omega*t + roll_phase) \
              + np.random.normal(0, gyro_noise_dps, n)
    gyro_y = -pitch_amp_deg_imu * omega * np.sin(omega*t + pitch_phase) \
              + np.random.normal(0, gyro_noise_dps, n)
    gyro_z = yaw_drift_rate * np.ones(n) \
              + np.random.normal(0, gyro_noise_dps, n)

    # Accelerometer readings: gravity projected onto body frame + heave on z
    accel_x_imu = -np.sin(pitch_rad)
    accel_y_imu = np.sin(roll_rad) * np.cos(pitch_rad)
    accel_z_imu = np.cos(roll_rad) * np.cos(pitch_rad)
    # add heave (vertical wave motion) on z
    heave_amp_g_imu = RAO_heave * A_wave * omega**2 / g_const
    accel_z_imu += -heave_amp_g_imu * np.sin(omega*t)
    accel_x_imu += np.random.normal(0, accel_noise_imu_g, n)
    accel_y_imu += np.random.normal(0, accel_noise_imu_g, n)
    accel_z_imu += np.random.normal(0, accel_noise_imu_g, n)

    # ---- Sensor fusion: try imufusion, fall back to complementary filter ----
    fusion_method = 'complementary'
    try:
        import imufusion
        ahrs = imufusion.Ahrs()
        # Configure: gain 0.5, gyro range 2000 dps, rejection 10/10,
        # recovery 5*fs samples
        ahrs.settings = imufusion.Settings(
            imufusion.CONVENTION_NWU,
            0.5, 2000, 10, 10, 5 * fs,
        )
        euler = np.empty((n, 3))
        dt = 1/fs
        for i in range(n):
            ahrs.update_no_magnetometer(
                np.array([gyro_x[i], gyro_y[i], gyro_z[i]]),
                np.array([accel_x_imu[i], accel_y_imu[i], accel_z_imu[i]]),
                dt,
            )
            euler[i] = ahrs.quaternion.to_euler()
        roll_fused_deg  = euler[:, 0]
        pitch_fused_deg = euler[:, 1]
        yaw_fused_deg   = euler[:, 2]
        fusion_method = 'imufusion'
    except ImportError:
        # Complementary filter fallback (pure numpy)
        #   roll_fused  = α·(prev_roll + gyro_x·dt) + (1-α)·accel_roll
        #   pitch_fused = α·(prev_pitch + gyro_y·dt) + (1-α)·accel_pitch
        #   yaw_fused   = integrate gyro_z (no absolute reference w/o magnetometer)
        alpha = 0.98
        dt = 1/fs
        # Tilt angles from accelerometer (radians, then to degrees)
        accel_roll_rad  = np.arctan2(accel_y_imu, accel_z_imu)
        accel_pitch_rad = np.arctan2(-accel_x_imu,
                                     np.sqrt(accel_y_imu**2 + accel_z_imu**2))
        accel_roll_deg  = np.rad2deg(accel_roll_rad)
        accel_pitch_deg = np.rad2deg(accel_pitch_rad)

        roll_fused_deg  = np.zeros(n)
        pitch_fused_deg = np.zeros(n)
        yaw_fused_deg   = np.zeros(n)
        for i in range(1, n):
            roll_fused_deg[i] = alpha * (roll_fused_deg[i-1] + gyro_x[i]*dt) \
                                 + (1-alpha) * accel_roll_deg[i]
            pitch_fused_deg[i] = alpha * (pitch_fused_deg[i-1] + gyro_y[i]*dt) \
                                  + (1-alpha) * accel_pitch_deg[i]
            yaw_fused_deg[i]   = yaw_fused_deg[i-1] + gyro_z[i]*dt

    # Fusion quality metrics
    roll_rms_err  = float(np.sqrt(np.mean((roll_fused_deg  - roll_true_deg )**2)))
    pitch_rms_err = float(np.sqrt(np.mean((pitch_fused_deg - pitch_true_deg)**2)))
    yaw_final_err = float(yaw_fused_deg[-1] - yaw_true_deg[-1])

    # =================================================================
    # Legacy gyro (pitch with linear bias drift) — kept for Section 6
    # =================================================================
    pitch_amp_deg = 5.0
    pitch_rate_true_dps = pitch_amp_deg * omega * np.cos(omega*t)
    bias_drift_dps      = 0.05 * t / duration
    pitch_rate_noisy    = pitch_rate_true_dps + bias_drift_dps \
                          + np.random.normal(0, 0.1, n)

    return {
        # ----- common -----
        't': t, 'fs': fs,
        'A_wave': A_wave, 'T_wave': T_wave, 'omega': omega,
        'wave_direction_deg': wave_direction_deg,
        'RAO_heave': RAO_heave, 'RAO_surge': RAO_surge, 'RAO_sway': RAO_sway,
        'phase_heave': phase_heave, 'phase_surge': phase_surge, 'phase_sway': phase_sway,
        'noise_g': noise_g,
        # ----- 3-axis acceleration (Block 1) -----
        'az_amp_g':  az_amp_g, 'az_amp_mps2': az_amp_mps2,
        'ax_amp_g':  ax_amp_g, 'ax_amp_mps2': ax_amp_mps2,
        'ay_amp_g':  ay_amp_g, 'ay_amp_mps2': ay_amp_mps2,
        'az_reading_g': az_reading_g,
        'ax_reading_g': ax_reading_g,
        'ay_reading_g': ay_reading_g,
        # ----- 4 anomaly patterns (Block 2) -----
        'signal_normal':    signal_normal,
        'signal_drift':     signal_drift,
        'signal_saturated': signal_saturated,
        'signal_stuck':     signal_stuck,
        'drift_rate_g_per_s': drift_rate_g_per_s,
        'drift_trend_line':   drift_trend_line,
        'storm_mult':         storm_mult,
        'sat_max': sat_max, 'sat_min': sat_min,
        'stuck_value':        stuck_value,
        # ----- IMU fusion (Block 3) -----
        'roll_amp_deg':      roll_amp_deg,
        'pitch_amp_deg_imu': pitch_amp_deg_imu,
        'roll_phase':        roll_phase,
        'pitch_phase':       pitch_phase,
        'yaw_drift_rate':    yaw_drift_rate,
        'gyro_noise_dps':    gyro_noise_dps,
        'accel_noise_imu_g': accel_noise_imu_g,
        'roll_true_deg':  roll_true_deg,
        'pitch_true_deg': pitch_true_deg,
        'yaw_true_deg':   yaw_true_deg,
        'gyro_x': gyro_x, 'gyro_y': gyro_y, 'gyro_z': gyro_z,
        'accel_x_imu': accel_x_imu, 'accel_y_imu': accel_y_imu,
        'accel_z_imu': accel_z_imu,
        'roll_fused_deg':  roll_fused_deg,
        'pitch_fused_deg': pitch_fused_deg,
        'yaw_fused_deg':   yaw_fused_deg,
        'fusion_method':   fusion_method,
        'roll_rms_err':    roll_rms_err,
        'pitch_rms_err':   pitch_rms_err,
        'yaw_final_err':   yaw_final_err,
        # ----- legacy gyro -----
        'pitch_rate_true':  pitch_rate_true_dps,
        'pitch_rate_noisy': pitch_rate_noisy,
        'bias_drift':       bias_drift_dps,
        'pitch_amp_deg':    pitch_amp_deg,
    }


def make_pressure_data(pto_baseline_bar=28.0, pto_swing_bar=2.0):
    """5 pressure sensors.

    Parameters (tunable via UI sliders):
      pto_baseline_bar — standing pressure in the PTO chamber (bar)
      pto_swing_bar    — wave-driven pressure swing amplitude (bar)
    """
    fs = 10
    duration = 60
    t = np.arange(0, duration, 1/fs)
    T_wave = 8.0
    omega = 2*np.pi/T_wave
    h = 50.0     # water depth
    k_wave = omega**2 / 9.8   # deep-water dispersion (approx)
    rho_g = 1025 * 9.8
    A = 1.0      # wave amplitude

    # Hydrostatic pressure on hull at 3 depths (1m, 3m, 6m)
    depths = [1.0, 3.0, 6.0]
    hull_pressures = {}
    for d in depths:
        decay = np.cosh(k_wave*(h-d)) / np.cosh(k_wave*h)
        P_dynamic = rho_g * A * np.cos(omega*t) * decay / 1e5   # bar
        P_static = rho_g * d / 1e5
        hull_pressures[f'd={d}m'] = P_static + P_dynamic + np.random.normal(0, 0.005, len(t))

    # Internal PTO pressure: baseline + swing at wave freq
    P_internal = pto_baseline_bar + pto_swing_bar * np.sin(omega*t + np.pi/4) \
                  + np.random.normal(0, 0.02, len(t))

    # Differential pressure across turbine (at wave frequency, but rectified somewhat)
    P_diff = 5.0 + 3.0 * np.abs(np.sin(omega*t)) + np.random.normal(0, 0.05, len(t))

    # Dry cabin: 1 atm = 1.013 bar, very stable
    cabin_baseline_bar = 1.013
    P_cabin = cabin_baseline_bar + np.random.normal(0, 0.0005, len(t))
    # Inject leak event at t > 40s
    leak_onset_s = 40.0
    leak_mask = t > leak_onset_s
    P_cabin[leak_mask] += (t[leak_mask] - leak_onset_s) * 0.001   # slow rise

    # Depth sensor at hull bottom: total pressure = static (ρ·g·d_draft) +
    # dynamic (same linear-wave-theory formula as Sensor 1, at z = draft)
    d_draft = 6.0  # hull-bottom depth below waterline, m
    decay_draft = np.cosh(k_wave*(h - d_draft)) / np.cosh(k_wave*h)
    P_static_draft = rho_g * d_draft / 1e5                            # bar
    P_dyn_draft    = rho_g * A * decay_draft * np.cos(omega*t) / 1e5  # bar
    P_depth = P_static_draft + P_dyn_draft + np.random.normal(0, 0.002, len(t))

    return {
        't': t, 'T_wave': T_wave, 'depths': depths,
        'hull': hull_pressures,
        'P_internal': P_internal, 'P_diff': P_diff,
        'P_cabin': P_cabin, 'P_depth': P_depth,
        'd_draft': d_draft,
        'P_static_draft': float(P_static_draft),
        'P_dyn_amplitude_bar': float(rho_g * A * decay_draft / 1e5),
        # PTO design parameters exposed for prose / plot annotations
        'pto_baseline_bar':  pto_baseline_bar,
        'pto_swing_bar':     pto_swing_bar,
        'cabin_baseline_bar': cabin_baseline_bar,
        'leak_onset_s':      leak_onset_s,
        # Wave parameters useful for the prose
        'A_wave':    A,
        'omega':     omega,
        'water_depth_h': h,
    }


def make_adcp_data(
    M2_amplitude=0.25, wind_surface=0.12, e_fold=8.0,
    bbl_thickness=5.0, surface_coupling=0.6,
):
    """ADCP velocity profile + GPS fusion.

    Parameters (all tunable via UI sliders):
      M2_amplitude     — M2 barotropic tidal amplitude (m/s)
      wind_surface     — wind-driven surface speed u₀ (m/s)
      e_fold           — wind-current e-folding depth scale L_e (m)
      bbl_thickness    — bottom-boundary-layer thickness (m)
      surface_coupling — float velocity / surface velocity ratio
    """
    duration_s = 24 * 3600              # 24 hours -> two full M2 cycles
    sample_period_s = 600               # 10-minute ADCP averages
    n = duration_s // sample_period_s   # 144 samples
    t_hr = np.arange(n) * sample_period_s / 3600.0

    # Geometry
    h_water = 50.0                      # local water depth
    adcp_depth = 10.0                   # transducer at hull bottom
    bin_thickness = 1.0
    n_bins = 30
    depths = adcp_depth + np.arange(1, n_bins + 1) * bin_thickness   # 11 -> 40 m

    # Component 1: M2 barotropic tide (uniform with depth, attenuated in BBL)
    M2_period_hr = 12.4
    M2_omega = 2 * np.pi / M2_period_hr
    u_M2_t = M2_amplitude * np.cos(M2_omega * t_hr)

    # Component 2: Wind-driven surface current with exponential decay
    wind_profile = wind_surface * np.exp(-depths / e_fold)            # by depth

    # Component 3: Bottom boundary layer (linear taper in bottom bbl_thickness m)
    bbl_factor = np.where(
        depths > (h_water - bbl_thickness),
        np.maximum((h_water - depths) / bbl_thickness, 0.0),
        1.0,
    )

    # Assemble u_true(t, z)
    u_true = np.zeros((n, n_bins))
    for i in range(n_bins):
        u_true[:, i] = u_M2_t * bbl_factor[i] + wind_profile[i]

    # Float (free-drifting) — surface coupling + extra wind drag
    wind_drag = 0.04
    u_surface_total = u_M2_t + wind_surface                          # surface current
    u_device = surface_coupling * u_surface_total + wind_drag        # 1D time series

    # Raw ADCP reading = absolute water current − device velocity (+ ADCP noise)
    np.random.seed(42)
    u_raw = u_true - u_device[:, np.newaxis] \
             + np.random.normal(0, 0.01, u_true.shape)

    # GPS estimate of device velocity (with realistic noise)
    gps_noise_m_s = 0.015                                            # 1.5 cm/s
    u_device_gps = u_device + np.random.normal(0, gps_noise_m_s, n)

    # Corrected current = raw + GPS-derived device velocity
    u_corrected = u_raw + u_device_gps[:, np.newaxis]

    # Fusion residual = corrected − truth (noise floor)
    u_residual = u_corrected - u_true
    fusion_residual_cm_s = float(np.std(u_residual)) * 100

    return {
        't_hr':           t_hr,
        'depths':         depths,
        'h_water':        h_water,
        'adcp_depth':     adcp_depth,
        'n_bins':         n_bins,
        'bin_thickness':  bin_thickness,
        'u_true':         u_true,
        'u_raw':          u_raw,
        'u_device':       u_device,
        'u_device_gps':   u_device_gps,
        'u_corrected':    u_corrected,
        'fusion_residual_cm_s': fusion_residual_cm_s,
        # Setup metadata for the UI
        'M2_amplitude':       M2_amplitude,
        'M2_period_hr':       M2_period_hr,
        'wind_surface':       wind_surface,
        'wind_e_fold':        e_fold,
        'bbl_thickness':      bbl_thickness,
        'surface_coupling':   surface_coupling,
        'wind_drag':          wind_drag,
        'gps_noise_cm_s':     gps_noise_m_s * 100,
        'duration_hr':        duration_s / 3600,
        'sample_period_min':  sample_period_s / 60,
        'n_samples':          n,
    }


# Load all data at startup
print("Pre-computing sensor data...")
DATA = {
    'fatigue': make_fatigue_data(),
    'strain': make_strain_data(),
    'motion': make_motion_data(),
    'pressure': make_pressure_data(),
    'adcp': make_adcp_data(),
}
print("Data loaded.")


# ========================================================================
# REUSABLE UI COMPONENTS
# ========================================================================
def sidebar(title_text, what_md, concepts_md, params_dict):
    """Standard left-column sidebar: title, what's plotted, concepts, params.
    Empty concepts_md or params_dict will be omitted (no empty headers).
    Both what_md and concepts_md support inline MathJax via $...$."""
    sections = []

    # 'What is plotted'
    if what_md and what_md.strip():
        sections.append(html.Div([
            html.Div('What is plotted', style=STYLES['sidebar_h']),
            dcc.Markdown(what_md, mathjax=True, style=STYLES['sidebar_body']),
        ], style=STYLES['sidebar_section']))

    # 'Concepts' (only if present)
    if concepts_md and concepts_md.strip():
        sections.append(html.Div([
            html.Div('Concepts', style=STYLES['sidebar_h']),
            dcc.Markdown(concepts_md, mathjax=True, style=STYLES['sidebar_body']),
        ], style=STYLES['sidebar_section']))

    # 'Parameters' (only if present)
    if params_dict:
        param_rows = [
            html.Tr([
                html.Td(k, style=STYLES['param_label']),
                html.Td(v, style=STYLES['param_value']),
            ]) for k, v in params_dict.items()
        ]
        sections.append(html.Div([
            html.Div('Parameters', style=STYLES['sidebar_h']),
            html.Table(param_rows, style=STYLES['param_table']),
        ], style=STYLES['sidebar_section']))

    return html.Div(sections, style=STYLES['sidebar'])


def tab_header(title, subtitle):
    """Standard tab header.

    `subtitle` can be a plain string (rendered as a paragraph) or a
    Markdown string with bullets / formatting (rendered via dcc.Markdown
    with MathJax enabled so inline math like $\\omega_n$ works).
    """
    if isinstance(subtitle, str) and ('\n-' in subtitle or '\n*' in subtitle
                                        or subtitle.lstrip().startswith('-')):
        subtitle_node = dcc.Markdown(subtitle,
                                       mathjax=True,
                                       style=STYLES['tab_subtitle'])
    else:
        subtitle_node = html.P(subtitle, style=STYLES['tab_subtitle'])
    return html.Div([
        html.H2(title, style=STYLES['tab_header']),
        subtitle_node,
    ])


# ========================================================================
# TAB BUILDERS  (each returns the content for one tab)
# ========================================================================

def tab_overview():
    """Tab 0: float schematic + sensor inventory with inline diagrams.
    No product names — generic 'wave energy converter' (WEC) framing.
    """
    # === Schematic figure (kept from before) ===
    fig = go.Figure()
    fig.add_shape(type='rect', x0=-3, y0=-8, x1=3, y1=0,
                  fillcolor=COLORS['water'], line=dict(width=0), opacity=0.4,
                  layer='below')
    fig.add_shape(type='line', x0=-3, y0=0, x1=3, y1=0,
                  line=dict(color=COLORS['accent'], width=2))
    fig.add_annotation(x=-2.8, y=0.2, text='water line', showarrow=False,
                       font=dict(size=11, color=COLORS['accent']),
                       xanchor='left')
    fig.add_shape(type='rect', x0=-1, y0=0, x1=1, y1=1.8,
                  fillcolor='#8E8A7B', line=dict(color=COLORS['text'], width=2))
    fig.add_annotation(x=0, y=0.9, text='TOP<br>BODY', showarrow=False,
                       font=dict(size=12, color='white', family=FONT))
    fig.add_shape(type='rect', x0=-0.3, y0=-7, x1=0.3, y1=0,
                  fillcolor='#6B6B6B', line=dict(color=COLORS['text'], width=2))
    fig.add_annotation(x=0.9, y=-3.5, text='NECK<br>(submerged)',
                       showarrow=False, font=dict(size=10), xanchor='left')
    fig.add_shape(type='rect', x0=-0.8, y0=-7.7, x1=0.8, y1=-7,
                  fillcolor='#404040', line=dict(color=COLORS['text'], width=2))
    fig.add_annotation(x=0, y=-7.35, text='INERTIA MASS', showarrow=False,
                       font=dict(size=10, color='white'))
    fig.add_annotation(x=0, y=1.8, ax=0, ay=3.0,
                       xref='x', yref='y', axref='x', ayref='y',
                       text='F(t) wave force',
                       showarrow=True, arrowhead=3, arrowsize=1.5,
                       arrowwidth=2, arrowcolor=COLORS['alert'],
                       font=dict(size=11, color=COLORS['alert']),
                       xanchor='center')
    sensors = [
        (0,  0.9,  'A',   'Accel/IMU'),
        (0.6, -0.5,  'S',   'Strain'),
        (-0.6, -0.5, 'S',   'Strain'),
        (0,  -3,   'P',   'Pressure'),
        (-1, 0.2, 'G',   'GPS'),
        (1,  -7.3, 'T',   'Temp'),
        (-1, -3,  'AD',  'ADCP'),
    ]
    for x, y, sym, lbl in sensors:
        fig.add_annotation(x=x, y=y, text=sym, showarrow=False,
                           font=dict(size=11, color='white', family=FONT),
                           bgcolor=COLORS['accent'], borderpad=3,
                           borderwidth=0)

    fig.update_layout(
        plot_layout(
            xaxis=dict(visible=False, range=[-3.5, 3.5]),
            yaxis=dict(visible=False, range=[-8.5, 4],
                       scaleanchor='x', scaleratio=1),
            margin=dict(l=10, r=10, t=10, b=10),
            height=520,
            showlegend=False,
        )
    )

    # === LEFT column: schematic + Architecture legend + wave-energy intro ===
    left = html.Div([
        dcc.Graph(figure=fig, config={'displayModeBar': False}),
        # Architecture legend RIGHT BELOW the figure (like a figure caption)
        html.Div([
            html.Div('Architecture — what you\'re looking at',
                      style={**STYLES['sidebar_h'], 'fontSize': '12px',
                              'marginTop': '0px'}),
            dcc.Markdown(
                "- **Top body** (above water) rides the wave\n"
                "- **Inertia mass** (below) stays relatively still\n"
                "- **Neck** between them connects through a power take-off (PTO)\n"
                "- **Relative motion** between top and bottom drives the PTO → "
                "electricity\n"
                "- **Critical structural feature:** the neck weld just below the "
                "waterline — bends with every wave, accumulates fatigue damage "
                "over the 20-year design life",
                style={**STYLES['sidebar_body'], 'fontSize': '12.5px'},
            ),
        ], style={**STYLES['sidebar_section'], 'paddingTop': '14px',
                   'paddingBottom': '14px'}),

        html.Div([
            html.Div('Why wave energy', style=STYLES['sidebar_h']),
            dcc.Markdown(
                "A comparison most people haven't seen:\n\n"
                "- Every square meter of solar panel receives about "
                "**0.2-0.3 kW** of solar energy.\n"
                "- Every meter of wind tower height absorbs roughly "
                "**2-3 kW**.\n"
                "- **Every meter** of California coastline receives "
                "**~30 kW of wave energy.**\n\n"
                "An order of magnitude more energy density than wind, two "
                "orders more than solar. But: harvesting it requires "
                "structures that survive years of relentless cyclic loading "
                "in seawater.\n\n"
                "This dashboard demonstrates the sensor-data pipeline that "
                "monitors a fleet of **wave energy converters (WECs)** in "
                "operation — physics-based simulation, fault injection, and "
                "detection across motion, strain, pressure, currents, "
                "temperature, and GPS.",
                style=STYLES['sidebar_body'],
            ),
        ], style=STYLES['sidebar_section']),
    ], style=STYLES['sidebar'])

    # === RIGHT column: bullet sensors with inline diagrams ===
    # Each sensor block: badge + name + bullets + (optional inline diagram)
    def sensor_section(badge, name, bullets, inline_figure=None):
        items = []
        # Header row
        items.append(html.Div([
            html.Span(badge, style={
                'display':         'inline-block',
                'minWidth':        '40px',
                'padding':         '4px 12px',
                'backgroundColor': COLORS['accent'],
                'color':           'white',
                'fontFamily':      FONT,
                'fontWeight':      '600',
                'fontSize':        '13px',
                'borderRadius':    '4px',
                'textAlign':       'center',
                'marginRight':     '14px',
                'verticalAlign':   'middle',
            }),
            html.Span(name, style={
                'fontSize':       '16px',
                'fontWeight':     '600',
                'color':          COLORS['text'],
                'verticalAlign':  'middle',
            }),
        ], style={'marginBottom': '10px', 'marginTop': '8px'}))
        # Bullets — use dcc.Markdown so **bold** renders properly
        items.append(html.Div(
            dcc.Markdown(
                '\n'.join(f'- {b}' for b in bullets),
                style={
                    'fontSize':   '13.5px',
                    'lineHeight': '1.6',
                    'color':      COLORS['text'],
                },
            ),
            style={
                'marginLeft':   '40px',
                'marginBottom': '12px',
            },
        ))
        # Inline diagram (if provided)
        if inline_figure is not None:
            items.append(html.Div(
                dcc.Graph(figure=inline_figure, config={'displayModeBar': False}),
                style={'marginLeft': '40px', 'marginBottom': '24px'},
            ))
        return html.Div(items, style={
            'marginBottom': '8px',
            'paddingBottom': '8px',
            'borderBottom': f"1px solid {COLORS['border']}",
        })

    right = html.Div([
        html.Div('Sensor inventory', style={
            **STYLES['sidebar_h'],
            'fontSize': '15px',
            'margin': '4px 0 20px 0',
        }),

        sensor_section('A',
            'Accelerometer + Inertial Measurement Unit (IMU)',
            ['Measures **linear acceleration** in three axes (surge, sway, '
              'heave — the translational degrees of freedom).',
              'Combined with a **gyroscope** measuring angular rate around '
              'three axes (roll, pitch, yaw — the rotational degrees of freedom).'],
            inline_figure=make_6dof_figure(),
        ),

        sensor_section('S',
            'Strain gauges',
            ['Measure microscopic stretching/compression at the **neck weld** '
              'during wave-driven bending.',
              'Industry practice: place a **3-gauge rosette** at 0°, 45°, 90° '
              'and compute principal stress — direction-invariant fatigue input.'],
            inline_figure=make_neck_bending_figure(),
        ),

        sensor_section('P',
            'Pressure sensors — 5 channels',
            ['**External hull** at three depths (1, 3, 6 m) — wave loading input.',
              '**Internal PTO chamber** — power-stroke pressure.',
              '**Differential across turbine** — drives flow rate → RPM → power.',
              '**Dry cabin** (1 atm nominal) — leak detection.',
              '**Depth / draft sensor** at hull bottom — mooring drift check.'],
            inline_figure=make_pressure_schematic(),
        ),

        sensor_section('AD',
            'ADCP — Acoustic Doppler Current Profiler',
            ['**What it measures.** Horizontal water current velocity at '
              'multiple depths below the float (a *vertical profile* of the '
              'current).',
              '**How.** Pings short acoustic pulses downward and listens for '
              'the Doppler frequency shift in the echo from particles in the '
              'water — the shift gives the velocity component along each beam.',
              '**Configuration.** Hull-mounted, downward-looking, 4 angled '
              'beams (Janus configuration). Typical output: 20-30 depth bins '
              'spanning ~2-30 m below the float.'],
            inline_figure=make_adcp_schematic(),
        ),

        sensor_section('T',
            'Temperature — 5 channels',
            ['**Seawater** — external reference, the most stable channel.',
              '**Electronics enclosure** — internal-air temperature.',
              '**Battery pack** — safety-critical (charging is only safe '
              'within ~0-45 °C).',
              '**Generator winding** — insulation has a thermal class limit '
              '(~130 °C).',
              '**PTO bearing** — mounted on the rotating assembly inside the '
              'hull, near the seal.'],
        ),

        sensor_section('G',
            'GPS',
            ['**What it measures.** Latitude and longitude of the float, '
              'sampled at ~1/min (the platform drifts slowly, ~30 mi/day).',
              '**Constraint.** Requires line-of-sight to satellites, so '
              'measurements are valid **only when the antenna is above the '
              'water surface**. Submergence in heavy seas → dropouts.',
              'Velocity is derived by differencing successive positions; this '
              'is what we subtract from the ADCP raw readings to recover '
              'absolute current.'],
        ),
    ])

    return html.Div([
        tab_header(
            'WEC Sensor Dashboard — overview',
            "Physics-based simulation prototype for a free-drifting "
            "wave-energy converter. This Overview introduces the float "
            "architecture and the six sensor categories on board; each "
            "of the other tabs takes one sensor or one analysis pipeline "
            "and walks through it end to end."
        ),
        html.Div([left, right], style=STYLES['two_col']),
    ])


def tab_motion():
    """Tab 1: Motion tab — wrapper with wave + RAO sliders, dynamic
    container for the parameter-dependent content."""
    d_init = DATA['motion']

    slider_box_style = {
        'background': '#F5F5F0',
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '4px',
        'padding': '14px 18px',
        'margin': '0 0 20px 0',
    }
    slider_label_style = {
        'fontSize': '12px', 'fontWeight': '600',
        'color': COLORS['text_dim'],
        'textTransform': 'uppercase',
        'letterSpacing': '0.04em',
        'margin': '0 0 6px 0',
    }

    def _slider_block(label_text, slider):
        return html.Div([
            html.Div(label_text, style=slider_label_style),
            slider,
        ], style={'flex': '1 1 0', 'minWidth': '170px'})

    slider_A   = dcc.Slider(id='mot-A',   min=0.5, max=3.0, step=0.5,
                              value=d_init['A_wave'],
                              marks={v: f'{v}' for v in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
                              tooltip={'placement': 'bottom', 'always_visible': False})
    slider_T   = dcc.Slider(id='mot-T',   min=4, max=16, step=2,
                              value=int(d_init['T_wave']),
                              marks={v: f'{v}' for v in [4, 6, 8, 10, 12, 14, 16]},
                              tooltip={'placement': 'bottom', 'always_visible': False})
    slider_th  = dcc.Slider(id='mot-theta', min=0, max=90, step=15,
                              value=int(d_init['wave_direction_deg']),
                              marks={v: f'{v}°' for v in [0, 30, 60, 90]},
                              tooltip={'placement': 'bottom', 'always_visible': False})
    slider_RAh = dcc.Slider(id='mot-RAOh', min=0.2, max=1.0, step=0.1,
                              value=d_init['RAO_heave'],
                              marks={v/10: f'{v/10:.1f}' for v in [2, 4, 6, 8, 10]},
                              tooltip={'placement': 'bottom', 'always_visible': False})
    slider_RAs = dcc.Slider(id='mot-RAOs', min=0.2, max=1.0, step=0.1,
                              value=d_init['RAO_surge'],
                              marks={v/10: f'{v/10:.1f}' for v in [2, 4, 6, 8, 10]},
                              tooltip={'placement': 'bottom', 'always_visible': False})
    slider_RAw = dcc.Slider(id='mot-RAOw', min=0.2, max=1.0, step=0.1,
                              value=d_init['RAO_sway'],
                              marks={v/10: f'{v/10:.1f}' for v in [2, 4, 6, 8, 10]},
                              tooltip={'placement': 'bottom', 'always_visible': False})

    # Phase shifts: 0 → 2π in π/4 steps  (8 stops; π marks displayed as fractions)
    PHASE_MARKS = {
        0: '0', np.pi/2: 'π/2', np.pi: 'π',
        3*np.pi/2: '3π/2', 2*np.pi: '2π',
    }
    slider_PhH = dcc.Slider(id='mot-PhH', min=0, max=2*np.pi, step=np.pi/4,
                              value=d_init['phase_heave'],
                              marks=PHASE_MARKS,
                              tooltip={'placement': 'bottom', 'always_visible': False})
    slider_PhS = dcc.Slider(id='mot-PhS', min=0, max=2*np.pi, step=np.pi/4,
                              value=d_init['phase_surge'],
                              marks=PHASE_MARKS,
                              tooltip={'placement': 'bottom', 'always_visible': False})
    slider_PhW = dcc.Slider(id='mot-PhW', min=0, max=2*np.pi, step=np.pi/4,
                              value=d_init['phase_sway'],
                              marks=PHASE_MARKS,
                              tooltip={'placement': 'bottom', 'always_visible': False})

    # Rotation amplitudes (deg)
    slider_RollA  = dcc.Slider(id='mot-rollA',  min=1, max=15, step=1,
                                  value=int(d_init['roll_amp_deg']),
                                  marks={v: f'{v}°' for v in [1, 5, 10, 15]},
                                  tooltip={'placement': 'bottom', 'always_visible': False})
    slider_PitchA = dcc.Slider(id='mot-pitchA', min=1, max=15, step=1,
                                  value=int(d_init['pitch_amp_deg_imu']),
                                  marks={v: f'{v}°' for v in [1, 5, 10, 15]},
                                  tooltip={'placement': 'bottom', 'always_visible': False})

    # Rotation phase shifts
    slider_RollPh  = dcc.Slider(id='mot-rollPh',  min=0, max=2*np.pi, step=np.pi/4,
                                  value=d_init['roll_phase'],
                                  marks=PHASE_MARKS,
                                  tooltip={'placement': 'bottom', 'always_visible': False})
    slider_PitchPh = dcc.Slider(id='mot-pitchPh', min=0, max=2*np.pi, step=np.pi/4,
                                  value=d_init['pitch_phase'],
                                  marks=PHASE_MARKS,
                                  tooltip={'placement': 'bottom', 'always_visible': False})

    # Yaw drift rate (deg/s)
    slider_yaw     = dcc.Slider(id='mot-yaw', min=0.0, max=0.20, step=0.02,
                                  value=d_init['yaw_drift_rate'],
                                  marks={v: f'{v:.2f}' for v in [0.0, 0.05, 0.10, 0.15, 0.20]},
                                  tooltip={'placement': 'bottom', 'always_visible': False})

    section_subhead = lambda txt: html.Div(txt, style={
        'fontSize': '11px', 'fontWeight': '700',
        'color': COLORS['text_dim'], 'textTransform': 'uppercase',
        'letterSpacing': '0.06em', 'margin': '14px 0 8px 0',
    })

    sliders_panel = html.Div([
        html.Div('Tune the wave field, the float\'s translation, and its '
                  'rotation — every signal, formula and amplitude updates '
                  'live', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text'],
            'margin': '0 0 4px 0',
        }),

        # === Wave field ===
        section_subhead('Wave field'),
        html.Div([
            _slider_block('Wave amplitude  A  (m)',    slider_A),
            _slider_block('Wave period  T  (s)',       slider_T),
            _slider_block('Incidence angle  θ  (deg)', slider_th),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}),

        # === Translation response (RAOs + phases) ===
        section_subhead('Float translation response  (heave / surge / sway)'),
        html.Div([
            _slider_block('Heave RAO',              slider_RAh),
            _slider_block('Surge RAO',              slider_RAs),
            _slider_block('Sway RAO',               slider_RAw),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}),
        html.Div([
            _slider_block('Heave phase  φ_heave  (rad)', slider_PhH),
            _slider_block('Surge phase  φ_surge  (rad)', slider_PhS),
            _slider_block('Sway phase  φ_sway  (rad)',   slider_PhW),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap',
                   'marginTop': '12px'}),

        # === Rotation response ===
        section_subhead('Float rotation response  (roll / pitch / yaw)'),
        html.Div([
            _slider_block('Roll amplitude  (deg)',  slider_RollA),
            _slider_block('Pitch amplitude (deg)',  slider_PitchA),
            _slider_block('Yaw drift rate (deg/s)', slider_yaw),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}),
        html.Div([
            _slider_block('Roll phase  φ_roll  (rad)',   slider_RollPh),
            _slider_block('Pitch phase  φ_pitch  (rad)', slider_PitchPh),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap',
                   'marginTop': '12px'}),
    ], style=slider_box_style)

    return html.Div([
        tab_header(
            'Motion — accelerometer and gyroscope signals',
            "- **Physics-based simulation** of the accelerometer and "
            "gyroscope readings produced by a wave-driven float.\n"
            "- **Sensor fusion** combining the two channels to recover "
            "the float's three-axis orientation.\n"
            "- **Fault detection** for four characteristic sensor "
            "failure modes (drift, saturation, stuck, dropout) with "
            "the signatures that distinguish them.",
        ),
        sliders_panel,
        dcc.Loading(
            id='motion-loading',
            type='circle',
            color=COLORS['accent'],
            children=html.Div(id='motion-output',
                                children=_tab_motion_body(d_init)),
        ),
    ])


def _tab_motion_body(d):
    """Build the parameter-dependent content of the Motion tab. Called
    from `tab_motion()` for the initial render and from
    `update_motion_output()` whenever a slider moves."""

    # ===== 3 separate accelerometer figures so each can have its own
    # description bullet directly below it. Formulas are in the main-text
    # bullets *above* each plot (in the right-column rendering), not in
    # the plot titles, so the titles stay short and readable. =====
    fig_az = go.Figure()
    fig_az.add_trace(go.Scatter(
        x=d['t'], y=d['az_reading_g'],
        line=dict(color=COLORS['accent_2'], width=1.0),
        showlegend=False,
    ))
    fig_az.add_hline(y=1.0, line_dash='dash', line_color=COLORS['text_dim'],
                      opacity=0.6,
                      annotation_text='1 g gravity baseline',
                      annotation_position='right')
    fig_az.update_layout(plot_layout(
        title=dict(text='a_z(t)  —  body-axis z accelerometer reading  (g)',
                    font=dict(size=12)),
        xaxis_title='Time (s)', yaxis_title='a_z (g)',
        yaxis=dict(range=[0.85, 1.15]),
        height=220,
        margin=dict(l=60, r=20, t=40, b=40),
    ))

    fig_ax = go.Figure()
    fig_ax.add_trace(go.Scatter(
        x=d['t'], y=d['ax_reading_g'],
        line=dict(color=COLORS['plot_1'], width=1.0),
        showlegend=False,
    ))
    fig_ax.add_hline(y=0, line_dash='dash', line_color=COLORS['text_dim'],
                      opacity=0.6)
    fig_ax.update_layout(plot_layout(
        title=dict(text='a_x(t)  —  body-axis x accelerometer reading  (g)',
                    font=dict(size=12)),
        xaxis_title='Time (s)', yaxis_title='a_x (g)',
        yaxis=dict(range=[-0.10, 0.10]),
        height=200,
        margin=dict(l=60, r=20, t=40, b=40),
    ))

    fig_ay = go.Figure()
    fig_ay.add_trace(go.Scatter(
        x=d['t'], y=d['ay_reading_g'],
        line=dict(color=COLORS['warning'], width=1.0),
        showlegend=False,
    ))
    fig_ay.add_hline(y=0, line_dash='dash', line_color=COLORS['text_dim'],
                      opacity=0.6)
    fig_ay.update_layout(plot_layout(
        title=dict(text='a_y(t)  —  body-axis y accelerometer reading  (g)',
                    font=dict(size=12)),
        xaxis_title='Time (s)', yaxis_title='a_y (g)',
        yaxis=dict(range=[-0.10, 0.10]),
        height=200,
        margin=dict(l=60, r=20, t=40, b=40),
    ))

    # ===== 4 separate anomaly figures, each with its own description bullet =====
    def _anom_fig(y, color, title, yrange, extra_traces=None, hline=None):
        f = go.Figure()
        f.add_trace(go.Scatter(x=d['t'], y=y, line=dict(color=color, width=1.0),
                                showlegend=False))
        for tr in (extra_traces or []):
            f.add_trace(tr)
        if hline is not None:
            for h in hline:
                f.add_hline(**h)
        f.update_layout(plot_layout(
            title=dict(text=title, font=dict(size=12)),
            xaxis_title='Time (s)', yaxis_title='a_z (g)',
            yaxis=dict(range=yrange),
            height=200,
            margin=dict(l=60, r=20, t=40, b=40),
        ))
        return f

    fig_anom_normal = _anom_fig(
        d['signal_normal'], COLORS['accent_2'],
        'a_z(t)  —  NORMAL  (healthy sensor)',
        [0.85, 1.30],
        hline=[dict(y=1.0, line_dash='dash', line_color=COLORS['text_dim'],
                     opacity=0.5)],
    )

    fig_anom_drift = _anom_fig(
        d['signal_drift'], COLORS['warning'],
        f"a_z(t)  —  DRIFT  (slow baseline shift, "
        f"+{d['drift_rate_g_per_s']*1000:.1f} mg/s)",
        [0.85, 1.30],
        extra_traces=[go.Scatter(
            x=d['t'], y=d['drift_trend_line'],
            line=dict(color=COLORS['warning'], width=1.5, dash='dot'),
            opacity=0.7, name='drifting baseline', showlegend=False,
        )],
    )

    fig_anom_sat = _anom_fig(
        d['signal_saturated'], COLORS['alert'],
        f"a_z(t)  —  SATURATION  (storm waves ×{d['storm_mult']:.0f}, "
        f"clipped at [{d['sat_min']:.2f}, {d['sat_max']:.2f}] g)",
        [0.70, 1.30],
        hline=[
            dict(y=d['sat_max'], line_dash='dash', line_color=COLORS['alert'],
                  opacity=0.5),
            dict(y=d['sat_min'], line_dash='dash', line_color=COLORS['alert'],
                  opacity=0.5),
        ],
    )

    fig_anom_stuck = _anom_fig(
        d['signal_stuck'], '#534AB7',
        f"a_z(t)  —  STUCK  (electronics frozen at {d['stuck_value']:.2f} g)",
        [0.85, 1.30],
        hline=[dict(y=d['stuck_value'], line_dash='dash',
                     line_color=COLORS['text_dim'], opacity=0.5)],
    )

    # ===== 3 separate IMU figures so each can have its own description =====
    # Figure: gyroscope readings — show the formulas with numbers substituted
    fig_gyro = go.Figure()
    for arr, name, color in [
        (d['gyro_x'], 'Gyro X  (roll rate  φ̇)',  COLORS['accent_2']),
        (d['gyro_y'], 'Gyro Y  (pitch rate θ̇)', COLORS['plot_1']),
        (d['gyro_z'], 'Gyro Z  (yaw rate   ψ̇)', COLORS['warning']),
    ]:
        fig_gyro.add_trace(go.Scatter(
            x=d['t'], y=arr, name=name,
            line=dict(color=color, width=0.9),
        ))
    fig_gyro.update_layout(plot_layout(
        title=dict(text='Gyroscope readings  —  body-axis angular rates (deg/s)',
                    font=dict(size=12)),
        xaxis_title='Time (s)', yaxis_title='Angular rate  (deg/s)',
        height=240,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation='h', y=-0.30, x=0.5, xanchor='center',
                     font=dict(size=10)),
    ))

    # Figure: accelerometer readings (gravity projection)
    fig_accel_imu = go.Figure()
    for arr, name, color in [
        (d['accel_x_imu'], 'Accel X  (-sin θ)',            COLORS['accent_2']),
        (d['accel_y_imu'], 'Accel Y  (sin φ·cos θ)',       COLORS['plot_1']),
        (d['accel_z_imu'], 'Accel Z  (cos φ·cos θ + heave)', COLORS['warning']),
    ]:
        fig_accel_imu.add_trace(go.Scatter(
            x=d['t'], y=arr, name=name,
            line=dict(color=color, width=0.9),
        ))
    fig_accel_imu.update_layout(plot_layout(
        title=dict(text='Accelerometer readings  —  gravity projection on body axes (g)',
                    font=dict(size=12)),
        xaxis_title='Time (s)', yaxis_title='Acceleration  (g)',
        height=240,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation='h', y=-0.30, x=0.5, xanchor='center',
                     font=dict(size=10)),
    ))

    # Figure: recovered orientation (fusion output) vs ground truth
    fig_orient = go.Figure()
    for true_arr, fused_arr, label, color in [
        (d['roll_true_deg'],  d['roll_fused_deg'],  'Roll  φ',  COLORS['accent_2']),
        (d['pitch_true_deg'], d['pitch_fused_deg'], 'Pitch θ',  COLORS['plot_1']),
        (d['yaw_true_deg'],   d['yaw_fused_deg'],   'Yaw   ψ',  COLORS['warning']),
    ]:
        fig_orient.add_trace(go.Scatter(
            x=d['t'], y=true_arr, name=f'True {label}',
            line=dict(color=color, width=1.8, dash='dash'),
        ))
        fig_orient.add_trace(go.Scatter(
            x=d['t'], y=fused_arr, name=f'Recovered {label}',
            line=dict(color=color, width=1.0),
            opacity=0.85,
        ))
    # Add an in-plot legend-style annotation with RMS errors
    rms_text = (
        f"Roll  φ:  RMS = {d['roll_rms_err']:.2f}°<br>"
        f"Pitch θ:  RMS = {d['pitch_rms_err']:.2f}°<br>"
        f"Yaw  ψ:  final err = {d['yaw_final_err']:+.2f}°"
    )
    fig_orient.add_annotation(
        text=rms_text,
        xref='paper', yref='paper',
        x=0.99, y=0.02,
        xanchor='right', yanchor='bottom',
        showarrow=False,
        font=dict(size=10, color=COLORS['text'], family=FONT),
        bgcolor='rgba(255,255,255,0.85)',
        bordercolor=COLORS['border'],
        borderwidth=1,
        borderpad=6,
    )
    fig_orient.update_layout(plot_layout(
        title=dict(text='Recovered orientation (fusion output) '
                         'vs ground truth  —  dashed = truth, solid = recovered',
                    font=dict(size=12)),
        xaxis_title='Time (s)', yaxis_title='Angle  (deg)',
        height=340,
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center',
                     font=dict(size=9)),
    ))

    # ============== Sidebar — concise sensor overview ==============
    side = sidebar(
        'Motion Sensors — accelerometer + gyro + IMU',
        """
**The three sensors on the float's motion channel:**

- **Accelerometer.** Measures **linear acceleration** on three body
  axes (a_x, a_y, a_z), in units of g or m/s². The vertical channel
  picks up the wave-driven heave; the two horizontals pick up surge
  (forward-back) and sway (side-to-side) — split by the wave
  direction angle.

- **Gyroscope.** Measures **angular velocity** on three body axes
  (rates of roll, pitch, yaw), in deg/s. Tells you how fast the
  device is rotating about each axis. The integral over time is
  the orientation — but raw integration drifts.

- **IMU (Inertial Measurement Unit).** The combined accel + gyro
  module (sometimes with a magnetometer). A fusion algorithm
  (Kalman / Madgwick) combines the channels into a drift-corrected
  orientation estimate.
""",
        "",
        {},
    )

    # ============== Main column ==============
    plots = html.Div([

        # =========== Section 1: wave kinematics ===========
        section_header('1. Wave kinematics — water motion at the float'),
        dcc.Markdown(
            "Start with a monochromatic wave at the float location. The "
            "**vertical** water-surface displacement and its time "
            "derivatives are:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'\eta(t) \;=\; A \sin(\omega t)'),
        latex_equation(r'\dot{\eta}(t) \;=\; A\,\omega \cos(\omega t)'),
        latex_equation(r'\ddot{\eta}(t) \;=\; -A\,\omega^{2} \sin(\omega t)'),

        dcc.Markdown(
            "In deep water, individual water particles **trace circles**: "
            "they go up, then forward, then down, then back. So the "
            "**horizontal** water-particle motion is 90° out of phase "
            "with the vertical motion. If the vertical position is "
            "$A\\sin(\\omega t)$, the horizontal position is "
            "$A\\cos(\\omega t)$, and the horizontal acceleration is "
            "$-A\\omega^{2}\\cos(\\omega t)$.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Markdown(
            "Waves rarely arrive head-on. They hit the float at some "
            "**incidence angle** $\\theta$ measured from the body x-axis. "
            "The horizontal motion projects onto body-x via $\\cos\\theta$ "
            "and onto body-y via $\\sin\\theta$.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Graph(figure=make_wave_direction_figure(d['wave_direction_deg']),
                   config={'displayModeBar': False}),

        dcc.Markdown(
            "An **accelerometer reads proper acceleration** — motion of "
            "the float plus the gravity vector. When the float is roughly "
            "level, gravity shows up as a constant $+g$ on body-z (and "
            "zero on body-x, body-y). The three body-axis accelerations "
            "are then:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'a_z(t) \;=\; -A\,\omega^{2} \sin(\omega t) \;+\; g'),
        latex_equation(r'a_x(t) \;=\; A\,\omega^{2} \cos\theta\,\cos(\omega t)'),
        latex_equation(r'a_y(t) \;=\; A\,\omega^{2} \sin\theta\,\cos(\omega t)'),

        styled_table(
            headers=['Parameter', 'Symbol', 'Value'],
            rows=[
                ['Wave amplitude',           'A',     f'{d["A_wave"]:.1f} m'],
                ['Wave period',              'T',     f'{d["T_wave"]:.1f} s'],
                ['Angular frequency',        'ω',     f'{d["omega"]:.3f} rad/s'],
                ['Wave incidence angle',     'θ',     f'{d["wave_direction_deg"]:.0f}° from body x-axis'],
                ['Gravity',                  'g',     '9.81 m/s²  =  1 g'],
            ],
        ),

        # =========== Section 2a: RAO ===========
        section_header('2a. Response Amplitude Operator — the response ratio'),
        dcc.Markdown(
            "Real floats don't follow waves 1:1. Each motion axis has a "
            "**Response Amplitude Operator (RAO)** — a dimensionless "
            "**transfer ratio** between 0 and 1 (or larger near resonance) "
            "that says how much of the wave amplitude the float captures "
            "on that axis. RAO is the industry-standard symbol for this "
            "in marine hydrodynamics.\n\n"
            f"For example, a heave RAO of {d['RAO_heave']:.2f} means: "
            f"when a 1 m wave passes the float, the float heaves with "
            f"{d['RAO_heave']:.2f} m of vertical motion. In production, "
            f"RAOs come from CFD simulations or wave-tank tests. Below "
            f"is a refresher on the six motion DOFs:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Graph(figure=make_6dof_figure(), config={'displayModeBar': False}),

        dcc.Markdown(
            "Adding the three RAOs to the Section 1 formulas:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'a_z(t) \;=\; -\,\text{RAO}_h \cdot A\,\omega^{2} \sin(\omega t) \;+\; g'),
        latex_equation(r'a_x(t) \;=\; \;\;\,\text{RAO}_s \cdot A\,\omega^{2} \cos\theta\,\cos(\omega t)'),
        latex_equation(r'a_y(t) \;=\; \;\;\,\text{RAO}_y \cdot A\,\omega^{2} \sin\theta\,\cos(\omega t)'),
        styled_table(
            headers=['Parameter', 'Symbol', 'Value'],
            rows=[
                ['Heave RAO',  'RAO_h', f'{d["RAO_heave"]:.2f}'],
                ['Surge RAO',  'RAO_s', f'{d["RAO_surge"]:.2f}'],
                ['Sway  RAO',  'RAO_y', f'{d["RAO_sway"]:.2f}'],
            ],
        ),

        # =========== Section 2b: Phase ===========
        section_header('2b. Phase shifts'),
        dcc.Markdown(
            "Even with RAOs known, the per-axis acceleration is not in "
            "phase with the wave — each axis has its own **phase shift** "
            "$\\varphi$ due to inertia, hydrodynamic damping, and "
            "structural coupling.\n\n"
            "Adding $\\varphi$ to each formula:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'a_z(t) \;=\; -\,\text{RAO}_h \cdot A\,\omega^{2} \sin(\omega t + \varphi_h) \;+\; g'),
        latex_equation(r'a_x(t) \;=\; \;\;\,\text{RAO}_s \cdot A\,\omega^{2} \cos\theta\,\cos(\omega t + \varphi_s)'),
        latex_equation(r'a_y(t) \;=\; \;\;\,\text{RAO}_y \cdot A\,\omega^{2} \sin\theta\,\cos(\omega t + \varphi_y)'),
        styled_table(
            headers=['Parameter', 'Symbol', 'Value'],
            rows=[
                ['Heave phase',  'φ_h',  f'{d["phase_heave"]:.1f} rad'],
                ['Surge phase',  'φ_s',  f'{d["phase_surge"]:.1f} rad'],
                ['Sway  phase',  'φ_y',  f'{d["phase_sway"]:.1f} rad'],
            ],
        ),

        # =========== Section 3: Derived peak amplitudes ===========
        section_header('3. Derived peak amplitudes'),
        dcc.Markdown(
            "Substituting the numbers into the peak-amplitude expressions:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        styled_table(
            headers=['Axis', 'Formula', 'Value'],
            rows=[
                ['a_z peak  (heave)',
                  'RAO_h · A · ω²',
                  f'{d["az_amp_mps2"]:.3f} m/s²  =  {d["az_amp_g"]*1000:.0f} mg'],
                ['a_x peak  (surge)',
                  'RAO_s · A · ω² · cos(θ)',
                  f'{d["ax_amp_mps2"]:.3f} m/s²  =  {d["ax_amp_g"]*1000:.0f} mg'],
                ['a_y peak  (sway)',
                  'RAO_y · A · ω² · sin(θ)',
                  f'{d["ay_amp_mps2"]:.3f} m/s²  =  {d["ay_amp_g"]*1000:.0f} mg'],
            ],
        ),

        # =========== Section 5: accelerometer plots ===========
        section_header('4. The accelerometer signals'),
        dcc.Markdown(
            "These three signals are computed directly from the Section 2b "
            "formulas, using the parameters above — plus a small amount of "
            "MEMS-grade sensor noise. Both axes are shown **normalised by "
            "$g$** (so the unit on the y-axis is dimensionless, with 1.0 = "
            "1 g = 9.81 m/s²).",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        dcc.Markdown(
            f"$a_z(t) = -\\text{{RAO}}_h\\cdot A\\cdot\\omega^2\\cdot"
            f"\\sin(\\omega t + \\varphi_h) + g$  "
            f"$= -{d['RAO_heave']:.2f}\\cdot {d['A_wave']:.1f}\\cdot "
            f"{d['omega']:.3f}^2\\cdot\\sin({d['omega']:.3f}\\,t + "
            f"{d['phase_heave']:.1f}) + g$",
            mathjax=True,
            style={'fontSize': '13px', 'lineHeight': '1.65',
                    'margin': '18px 0 4px 0'},
        ),
        dcc.Graph(figure=fig_az, config={'displayModeBar': False}),
        dcc.Markdown(
            f"Oscillates around the **1 g gravity baseline** (dashed line). "
            f"Peak wave-driven motion ≈ ±{d['az_amp_g']*1000:.0f} mg around "
            f"the baseline.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        dcc.Markdown(
            f"$a_x(t) = \\text{{RAO}}_s\\cdot A\\cdot\\omega^2\\cdot"
            f"\\cos\\theta\\cdot\\cos(\\omega t + \\varphi_s)$  "
            f"$= {d['RAO_surge']:.2f}\\cdot {d['A_wave']:.1f}\\cdot "
            f"{d['omega']:.3f}^2\\cdot\\cos({d['wave_direction_deg']:.0f}°)"
            f"\\cdot\\cos({d['omega']:.3f}\\,t + {d['phase_surge']:.1f})$",
            mathjax=True,
            style={'fontSize': '13px', 'lineHeight': '1.65',
                    'margin': '14px 0 4px 0'},
        ),
        dcc.Graph(figure=fig_ax, config={'displayModeBar': False}),
        dcc.Markdown(
            f"Oscillates around **zero** (no gravity baseline on x). Peak "
            f"≈ ±{d['ax_amp_g']*1000:.0f} mg.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        dcc.Markdown(
            f"$a_y(t) = \\text{{RAO}}_y\\cdot A\\cdot\\omega^2\\cdot"
            f"\\sin\\theta\\cdot\\cos(\\omega t + \\varphi_y)$  "
            f"$= {d['RAO_sway']:.2f}\\cdot {d['A_wave']:.1f}\\cdot "
            f"{d['omega']:.3f}^2\\cdot\\sin({d['wave_direction_deg']:.0f}°)"
            f"\\cdot\\cos({d['omega']:.3f}\\,t + {d['phase_sway']:.1f})$",
            mathjax=True,
            style={'fontSize': '13px', 'lineHeight': '1.65',
                    'margin': '14px 0 4px 0'},
        ),
        dcc.Graph(figure=fig_ay, config={'displayModeBar': False}),
        dcc.Markdown(
            f"Oscillates around **zero**. Peak ≈ "
            f"±{d['ay_amp_g']*1000:.0f} mg — smaller than a_x because "
            f"sin {d['wave_direction_deg']:.0f}° = {math.sin(math.radians(d['wave_direction_deg'])):.2f} < "
            f"cos {d['wave_direction_deg']:.0f}° = {math.cos(math.radians(d['wave_direction_deg'])):.2f}, "
            f"and the sway RAO ({d['RAO_sway']:.2f}) is smaller than the "
            f"surge RAO ({d['RAO_surge']:.2f}).",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # =========== Section 5: 4 anomaly patterns ===========
        section_header('5. Anomaly patterns'),
        dcc.Markdown(
            "The signals above assume a healthy sensor. In a deployed "
            "fleet, **any continuous sensor** (acceleration, strain, "
            "pressure, temperature, …) can fail in one of four "
            "characteristic ways. Here are those patterns applied to "
            "the vertical acceleration $a_z$ — the visual signatures and "
            "response actions are the same on every channel:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        dcc.Graph(figure=fig_anom_normal, config={'displayModeBar': False}),
        dcc.Markdown(
            "The actual $a_z$ signal from Section 4, smooth oscillation at "
            "the wave frequency around the 1 g gravity baseline. "
            "**Action:** none, sensor is healthy.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        dcc.Graph(figure=fig_anom_drift, config={'displayModeBar': False}),
        dcc.Markdown(
            f"Same signal with a slow baseline shift of "
            f"{d['drift_rate_g_per_s']*1000:.1f} mg/s — caused by hardware "
            f"aging, temperature change, or calibration loss. "
            f"**Action:** schedule recalibration.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        dcc.Graph(figure=fig_anom_sat, config={'displayModeBar': False}),
        dcc.Markdown(
            f"Storm waves (×{d['storm_mult']:.0f} normal amplitude) exceed "
            f"the sensor's measurement range — the signal clips at "
            f"[{d['sat_min']:.2f}, {d['sat_max']:.2f}] g. **Action:** "
            f"check whether the seas are unusually large *(real)* or the "
            f"sensor range is set too small *(hardware)*.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        dcc.Graph(figure=fig_anom_stuck, config={'displayModeBar': False}),
        dcc.Markdown(
            f"Sensor electronics dead — reading frozen at "
            f"{d['stuck_value']:.2f} g with only thermal noise. "
            f"**Action:** hardware ticket for replacement.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # =========== Section 6: IMU + sensor fusion ===========
        section_header('6. IMU + sensor fusion — recovering orientation'),

        dcc.Markdown(
            "The float doesn't just heave up and down. It also **tilts** "
            "and **rotates** about three body axes. To track that motion, "
            "we use the three Euler angles from the bottom row of the "
            "6-DOF figure in Section 2a:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Markdown(
            "- **Roll** $\\phi$ — rotation about the body x-axis (side-to-side tilt)\n"
            "- **Pitch** $\\theta$ — rotation about the body y-axis (nose up/down)\n"
            "- **Yaw** $\\psi$ — rotation about the body z-axis (heading change)",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ----- 6a: orientation model -----
        section_header('6a. Ground-truth orientation model'),
        dcc.Markdown(
            "We model the float's true orientation as a sinusoidal "
            "wave-driven roll and pitch, plus a slow linear yaw drift "
            "from asymmetric currents:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'\phi(t)   \;=\; \phi_{\text{amp}}   \cdot \sin(\omega t + \varphi_{\phi})'),
        latex_equation(r'\theta(t) \;=\; \theta_{\text{amp}} \cdot \cos(\omega t + \varphi_{\theta})'),
        latex_equation(r'\psi(t)   \;=\; \dot{\psi}_{\text{drift}} \cdot t'),
        styled_table(
            headers=['Parameter', 'Symbol', 'Value'],
            rows=[
                ['Roll  amplitude',   'φ_amp',  f'{d["roll_amp_deg"]:.1f}°'],
                ['Roll  phase',       'φ_φ',    f'{d["roll_phase"]:.2f} rad'],
                ['Pitch amplitude',   'θ_amp',  f'{d["pitch_amp_deg_imu"]:.1f}°'],
                ['Pitch phase',       'φ_θ',    f'{d["pitch_phase"]:.2f} rad'],
                ['Yaw drift rate',    'ψ̇',     f'{d["yaw_drift_rate"]:.2f} deg/s  (≈ 3 deg/min)'],
            ],
        ),

        # ----- 6b: gyro = derivative -----
        section_header('6b. Gyroscope readings — time derivative of the angles'),
        dcc.Markdown(
            "The gyroscope measures **angular velocity** — the time "
            "derivative of each Euler angle. Differentiating the "
            "expressions above:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'\dot{\phi}(t)   \;=\;  \phi_{\text{amp}}  \cdot \omega \cdot \cos(\omega t + \varphi_{\phi})'),
        latex_equation(r'\dot{\theta}(t) \;=\; -\theta_{\text{amp}} \cdot \omega \cdot \sin(\omega t + \varphi_{\theta})'),
        latex_equation(r'\dot{\psi}(t)   \;=\; \dot{\psi}_{\text{drift}}'),
        dcc.Markdown(
            f"Then we add **Gaussian white noise** at standard deviation "
            f"σ = **{d['gyro_noise_dps']:.1f} deg/s** — typical for a "
            f"MEMS-grade consumer gyroscope.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ----- 6c: why gyro alone fails -----
        section_header('6c. Why integrating the gyro alone fails'),
        dcc.Markdown(
            "Tempting reasoning: if we know angular velocity and an "
            "initial orientation, we can recover orientation by "
            "integration:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'\phi(t) \;=\; \phi_{0} \;+\; \int_{0}^{t} \dot{\phi}(\tau)\, d\tau'),
        dcc.Markdown(
            "Mathematically true, physically broken. Every real gyro has "
            "a small **bias** — a non-zero output when the device is "
            "actually still — that drifts slowly with temperature. "
            "Integrating a constant bias of even 0.05 deg/s for "
            "17 minutes produces a **50-degree orientation error**. "
            "By tomorrow the roll estimate is meaningless.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ----- 6d: accel provides gravity reference -----
        section_header('6d. The accelerometer provides a gravity reference'),
        dcc.Markdown(
            "We need an **absolute** reference to anchor the drifting "
            "gyro integration. The accelerometer provides one: gravity "
            "always points down in the world frame, so its components "
            "on the tilted body axes tell us the tilt directly.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        html.Div([
            html.Img(
                src=(f'data:image/jpeg;base64,{GRAVITY_B64}'
                      if GRAVITY_B64 else ''),
                style={'maxWidth': '100%', 'height': 'auto',
                        'display': 'block', 'margin': '8px auto'},
            ),
            html.Div(
                "Source: en2301.blogspot.com/2015/01/212-214.html",
                style={'fontSize': '11px', 'fontStyle': 'italic',
                        'color': COLORS['text_dim'], 'textAlign': 'center',
                        'marginTop': '4px'},
            ),
        ]),
        dcc.Markdown(
            "Decomposing a 3D vector $\\vec{F}$ into rectangular "
            "components is the same problem we have with gravity in the "
            "body frame. Applying the same geometry — when the float is "
            "level, gravity in the world frame is "
            "$\\vec{g}_{\\text{world}} = (0,\\,0,\\,g)$. After tilting by "
            "roll $\\phi$ and pitch $\\theta$, that same vector projects "
            "onto body axes as:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'a_x^{\text{grav}} \;=\; -g\,\sin\theta'),
        latex_equation(r'a_y^{\text{grav}} \;=\; \;\;\,g\,\sin\phi \cos\theta'),
        latex_equation(r'a_z^{\text{grav}} \;=\; \;\;\,g\,\cos\phi \cos\theta'),
        dcc.Markdown(
            f"Then we add **Gaussian white noise** at "
            f"σ = **{d['accel_noise_imu_g']*1000:.0f} mg** "
            f"— typical for a MEMS-grade consumer accelerometer.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ----- 6e: fusion + IMU concept -----
        section_header('6e. Sensor fusion — combining gyro + accelerometer'),
        dcc.Markdown(
            "An **IMU (Inertial Measurement Unit)** is the combined "
            "accelerometer + gyroscope module (sometimes with a "
            "magnetometer). Each sensor on its own is incomplete; a "
            "**sensor-fusion algorithm** (Kalman, Madgwick) combines "
            "them into a drift-corrected orientation estimate:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Markdown(
            "- **Gyroscope** — fast and accurate over short windows, "
            "but drifts without bound when integrated.\n"
            "- **Accelerometer** — gives the gravity direction, so it's "
            "**absolute** for roll and pitch; but noisy at short timescales.\n"
            "- **Magnetometer** *(optional)* — gives the only absolute "
            "yaw reference. Easily disturbed by nearby metal / current loops.\n"
            "- **Fusion** uses each sensor in the frequency band where "
            "it performs best: gyro at high frequencies, accelerometer at "
            "low frequencies, magnetometer for absolute yaw.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        expandable_note('What is a complementary filter', [
            dcc.Markdown(
                "A **complementary filter** is the simplest sensor-fusion "
                "scheme. Two reference signals give the same quantity "
                "(orientation) with opposite noise profiles:\n\n"
                "- The **gyro** is accurate at **high frequency** (fast "
                "rotations) but drifts at low frequency.\n"
                "- The **accelerometer** is accurate at **low frequency** "
                "(gravity always points down, on average) but noisy at "
                "high frequency.\n\n"
                "The filter trusts each sensor in its strong band and "
                "blends them with a weighted sum:\n\n"
                "$$\\phi_{\\text{fused}}(t) \\;=\\; \\alpha \\cdot \\big(\\phi_{\\text{fused}}(t-\\Delta t) + \\dot{\\phi}_{\\text{gyro}} \\Delta t\\big) \\;+\\; (1 - \\alpha) \\cdot \\phi_{\\text{accel}}$$\n\n"
                "with $\\alpha$ near 1 (typically 0.98). It's a first-order "
                "low-pass on the accelerometer + first-order high-pass on "
                "the gyro, combined. Madgwick / Mahony / Kalman filters "
                "are more sophisticated quaternion-based versions of the "
                "same idea.",
                mathjax=True,
                style={'fontSize': '13px', 'lineHeight': '1.65'},
            ),
        ]),
        dcc.Markdown(
            f"This dashboard uses the **imufusion** library "
            f"(Madgwick-filter implementation) when available, otherwise "
            f"a hand-coded complementary filter. Method used here: "
            f"**{d['fusion_method']}**.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ----- 6f: fusion results -----
        section_header('6f. Recovered orientation — feeding the noisy IMU '
                        'readings to the fusion filter'),
        dcc.Markdown(
            "Three steps: derive the gyroscope and accelerometer signals "
            "from the ground-truth angles in 6a, feed them to the fusion "
            "algorithm from 6e, and check how well the output recovers "
            "the truth.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ===== Gyroscope =====
        html.Div('Gyroscope inputs', style={
            'fontSize': '14px', 'fontWeight': '700',
            'color': COLORS['text'], 'margin': '20px 0 4px 0',
        }),
        dcc.Markdown(
            f"Differentiating the 6a angles (formulas from 6b) and adding "
            f"Gaussian white noise with σ = "
            f"**{d['gyro_noise_dps']:.1f} deg/s** ('RMS' = root-mean-"
            f"square, the standard deviation of the noise):\n\n"
            f"- $\\dot\\phi(t) = \\phi_{{\\text{{amp}}}}\\cdot\\omega\\cdot"
            f"\\cos(\\omega t + \\varphi_\\phi) + \\text{{noise}}$  "
            f"$= {d['roll_amp_deg']:.1f} \\cdot {d['omega']:.3f} \\cdot"
            f"\\cos({d['omega']:.3f}\\,t + {d['roll_phase']:.2f}) + "
            f"\\text{{noise}}$  →  peak "
            f"**±{d['roll_amp_deg']*d['omega']:.2f} deg/s**\n"
            f"- $\\dot\\theta(t) = -\\theta_{{\\text{{amp}}}}\\cdot\\omega\\cdot"
            f"\\sin(\\omega t + \\varphi_\\theta) + \\text{{noise}}$  "
            f"$= -{d['pitch_amp_deg_imu']:.1f} \\cdot {d['omega']:.3f} \\cdot"
            f"\\sin({d['omega']:.3f}\\,t + {d['pitch_phase']:.2f}) + "
            f"\\text{{noise}}$  →  peak "
            f"**±{d['pitch_amp_deg_imu']*d['omega']:.2f} deg/s**\n"
            f"- $\\dot\\psi(t) = \\dot\\psi_{{\\text{{drift}}}} + "
            f"\\text{{noise}} = {d['yaw_drift_rate']:.2f} \\text{{ deg/s}} + "
            f"\\text{{noise}}$  →  constant slow drift",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.85'},
        ),
        dcc.Graph(figure=fig_gyro, config={'displayModeBar': False}),
        dcc.Markdown(
            f"- Roll rate and pitch rate **oscillate at the wave "
            f"frequency** with the peaks listed above.\n"
            f"- Yaw rate sits at the **constant {d['yaw_drift_rate']:.2f} "
            f"deg/s** drift value, almost buried in the noise.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # ===== Accelerometer =====
        html.Div('Accelerometer inputs', style={
            'fontSize': '14px', 'fontWeight': '700',
            'color': COLORS['text'], 'margin': '24px 0 4px 0',
        }),
        dcc.Markdown(
            "**These are different from Section 4.** Section 4 modeled "
            "wave-driven acceleration on a *level* float. Here we model "
            "the **tilt-induced gravity projection** (6d) plus a small "
            "heave term on the z-channel — this is what the orientation "
            "filter actually needs.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Markdown(
            f"**Inputs to the formulas below:**\n\n"
            f"- $\\phi(t) = {d['roll_amp_deg']:.1f}°\\cdot\\sin"
            f"({d['omega']:.3f}\\,t)$  (from 6a roll)\n"
            f"- $\\theta(t) = {d['pitch_amp_deg_imu']:.1f}°\\cdot"
            f"\\cos({d['omega']:.3f}\\,t + {d['pitch_phase']:.2f})$  "
            f"(from 6a pitch)\n"
            f"- Gaussian white noise on each channel with σ = "
            f"**{d['accel_noise_imu_g']*1000:.0f} mg**\n\n"
            f"**Body-axis accelerations (in units of g):**\n\n"
            f"- $a_x(t) = -\\sin\\theta(t) + \\text{{noise}}$  →  peak "
            f"**≈ ±{math.sin(math.radians(d['pitch_amp_deg_imu']))*1000:.0f} mg** "
            f"($\\sin {d['pitch_amp_deg_imu']:.0f}°$)\n"
            f"- $a_y(t) = \\sin\\phi(t)\\cdot\\cos\\theta(t) + "
            f"\\text{{noise}}$  →  peak **≈ ±"
            f"{math.sin(math.radians(d['roll_amp_deg']))*math.cos(math.radians(d['pitch_amp_deg_imu']))*1000:.0f} mg** "
            f"($\\sin {d['roll_amp_deg']:.0f}° \\cdot \\cos {d['pitch_amp_deg_imu']:.0f}°$)\n"
            f"- $a_z(t) = \\cos\\phi(t)\\cdot\\cos\\theta(t) + "
            f"a_z^{{\\text{{heave}}}}(t) + \\text{{noise}}$  →  baseline "
            f"**≈ {math.cos(math.radians(d['roll_amp_deg']))*math.cos(math.radians(d['pitch_amp_deg_imu'])):.3f} g** "
            f"with heave swing "
            f"**±{(d['RAO_heave']*d['A_wave']*d['omega']**2/9.81)*1000:.0f} mg**",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.85'},
        ),
        dcc.Graph(figure=fig_accel_imu, config={'displayModeBar': False}),
        dcc.Markdown(
            "- **a_z** sits near **+1 g** (gravity baseline + small heave).\n"
            "- **a_x** and **a_y** oscillate around zero — the projection "
            "of gravity onto the tilted body axes, proportional to pitch "
            "and roll respectively.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # ===== Recovered orientation =====
        html.Div('Recovered orientation  (fusion output)', style={
            'fontSize': '14px', 'fontWeight': '700',
            'color': COLORS['text'], 'margin': '24px 0 4px 0',
        }),
        dcc.Markdown(
            "Feed the gyro and accelerometer time series into the fusion "
            "filter (imufusion / Madgwick, no magnetometer) — its output "
            "is the recovered orientation:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'\big[\phi_{\text{rec}}(t),\;\theta_{\text{rec}}(t),\;\psi_{\text{rec}}(t)\big] \;=\; \mathrm{Madgwick}\!\big(\,\dot{\phi},\,\dot{\theta},\,\dot{\psi},\;a_x,\,a_y,\,a_z\,\big)'
        ),
        dcc.Graph(figure=fig_orient, config={'displayModeBar': False}),
        dcc.Markdown(
            f"- **Roll and pitch** track the ground truth tightly — RMS "
            f"error ≈ **{max(d['roll_rms_err'], d['pitch_rms_err']):.2f}°**. "
            f"The accelerometer anchors them via gravity.\n"
            f"- **Yaw drifts away** from the ground truth — final-time "
            f"error ≈ **{d['yaw_final_err']:+.1f}°** after only 60 s. "
            f"The filter has no absolute heading reference (no "
            f"magnetometer), so it cannot trust the gyro_z integration "
            f"and the output slowly slips away from the true heading.\n"
            f"- Production IMUs add a **magnetometer** (or use GPS "
            f"heading) for absolute yaw reference. Without one, yaw is "
            f"the channel you can never fully trust.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 16px 0'},
        ),
    ], style=STYLES['plots'])

    return html.Div([side, plots], style=STYLES['two_col'])


def tab_fatigue():
    """Tab 3: Strain & Fatigue — wrapper with material sliders + scenario
    dropdown + one big dynamic content area (side + scenario output +
    all-scenarios comparison + summary table). The callback re-runs
    `make_fatigue_data` with new material parameters and rebuilds
    everything.
    """
    f_init = DATA['fatigue']
    mat_init = f_init['material']

    slider_box_style = {
        'background': '#F5F5F0',
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '4px',
        'padding': '14px 18px',
        'margin': '0 0 20px 0',
    }
    slider_label_style = {
        'fontSize': '12px', 'fontWeight': '600',
        'color': COLORS['text_dim'],
        'textTransform': 'uppercase',
        'letterSpacing': '0.04em',
        'margin': '0 0 6px 0',
    }
    def _slider_block(label, slider):
        return html.Div([
            html.Div(label, style=slider_label_style),
            slider,
        ], style={'flex': '1 1 0', 'minWidth': '170px'})

    # σ_limit (air) — BS 7608 class strength: 25, 56, 80, 100 MPa
    slider_sigLim = dcc.Slider(id='fat-sigma-lim-air', min=25, max=100, step=5,
                                  value=mat_init['sigma_limit_air'],
                                  marks={v: f'{v}' for v in [25, 56, 80, 100]},
                                  tooltip={'placement': 'bottom', 'always_visible': False})
    # Environment factor: 0.60, 0.75, 1.00
    slider_envF   = dcc.Slider(id='fat-env-factor', min=0.50, max=1.00, step=0.05,
                                  value=mat_init['env_factor'],
                                  marks={v/100: f'{v/100:.2f}' for v in [60, 75, 100]},
                                  tooltip={'placement': 'bottom', 'always_visible': False})
    # Yield strength — steel grades S235=250, S275=275, S355=355
    slider_yield  = dcc.Slider(id='fat-yield', min=200, max=450, step=25,
                                  value=mat_init['yield_str'],
                                  marks={v: f'{v}' for v in [200, 250, 350, 450]},
                                  tooltip={'placement': 'bottom', 'always_visible': False})
    # Ultimate tensile strength — S235=450, S355=510, high-strength=700+
    slider_UTS    = dcc.Slider(id='fat-UTS', min=400, max=700, step=25,
                                  value=mat_init['UTS'],
                                  marks={v: f'{v}' for v in [400, 500, 600, 700]},
                                  tooltip={'placement': 'bottom', 'always_visible': False})

    sliders_panel = html.Div([
        html.Div('Tune the material — every fatigue / yield / UTS '
                  'verdict updates live', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text'],
            'margin': '0 0 12px 0',
        }),
        html.Div([
            _slider_block('σ_limit  (in-air, MPa)',         slider_sigLim),
            _slider_block('Environment factor  (× σ_limit)', slider_envF),
            _slider_block('Yield strength σ_y  (MPa)',       slider_yield),
            _slider_block('Ultimate tensile σ_UTS  (MPa)',   slider_UTS),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}),
    ], style=slider_box_style)

    # Scenario dropdown lives OUTSIDE the dynamic area so its state
    # is preserved across material-slider changes.
    scenarios_list = list(f_init['scenarios'].keys())
    dropdown = html.Div([
        html.Label('Wave scenario:', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text'], 'marginRight': '12px',
            'verticalAlign': 'middle',
        }),
        dcc.Dropdown(
            id='fatigue-scenario',
            options=[{'label': s, 'value': s} for s in scenarios_list],
            value='Moderate',
            clearable=False,
            style={'width': '280px', 'display': 'inline-block',
                    'verticalAlign': 'middle'},
        ),
    ], style={'marginBottom': '20px'})

    return html.Div([
        tab_header(
            'Strain & Fatigue — will the float reach its 20-year design life?',
            "- **Fatigue-life analysis** of the critical welded neck "
            "joint over the 20-year design life, using rainflow cycle "
            "counting and an industry-standard offshore-welds fatigue "
            "curve.\n"
            "- **Monte Carlo simulation** of the peak-stress "
            "distribution, checked against the yield and ultimate-"
            "tensile-strength allowables with engineering safety "
            "factors.\n"
            "- **Verdict**: pass / fail and the remaining design margin "
            "for the selected sea state — switchable from calm sea to "
            "storm.",
        ),
        sliders_panel,
        dropdown,
        dcc.Loading(
            id='fatigue-loading',
            type='circle',
            color=COLORS['accent'],
            children=html.Div(id='fatigue-output',
                                children=_tab_fatigue_body(f_init, 'Moderate')),
        ),
    ])


def _tab_fatigue_sidebar(f):
    """Sidebar prose with material-dependent yield/UTS/σ_limit numbers."""
    mat = f['material']
    return sidebar(
        'Strain & Fatigue — design verification',
        f"""
**The headline question:** will the critical neck weld survive the
**20-year design life** under the wave loads it'll see?

**The design must pass three criteria simultaneously:**

1. **Fatigue.**  Cumulative damage $D < 1.0$ over the 20-year design
   life, where $D = \\sum n_i / N_i$ (Miner's rule).
2. **Yield.**  Peak observed stress $< \\sigma_y / \\text{{SF}}_y$, where
   $\\sigma_y = {mat['yield_str']:.0f}$ MPa and
   $\\text{{SF}}_y = {mat['yield_sf']}$.
3. **UTS.**  Peak observed stress $< \\sigma_{{\\text{{UTS}}}} /
   \\text{{SF}}_{{\\text{{UTS}}}}$, where
   $\\sigma_{{\\text{{UTS}}}} = {mat['UTS']:.0f}$ MPa and
   $\\text{{SF}}_{{\\text{{UTS}}}} = {mat['UTS_sf']}$.

Switch sea state with the dropdown above to see how each criterion's
margin changes.
""",
        f"""
**Hooke's law:** $\\sigma = E \\cdot \\varepsilon$. For steel
($E = 200$ GPa), 1 microstrain $\\approx$ 0.2 MPa. So a 100-microstrain
gauge reading = 20 MPa stress.

**BS 7608 S-N curve:** $N(\\sigma) = N_{{\\text{{lim}}}} \\cdot
(\\sigma_{{\\text{{lim}}}} / \\sigma)^{{m}}$
above the fatigue limit, infinite life below. With $m = 3$,
**doubling stress = 8× faster fatigue death**.

**Miner's rule:** $D = \\sum n_i / N_i$. Failure at $D = 1.0$.

**Environment factor:** offshore corrosion accelerates crack growth.
$\\sigma_{{\\text{{lim, seawater}}}} = {mat['env_factor']:.2f} \\cdot
\\sigma_{{\\text{{lim, air}}}}$
({mat['env_factor']*100:.0f}% of the in-air strength).

**Hard limits:** above yield → permanent deformation; above UTS →
fracture. Safety factors {mat['yield_sf']} (yield) and {mat['UTS_sf']} (UTS).
""",
        {},
    )


def _tab_fatigue_body(f, scenario_name):
    """Build the parameter-dependent content of the Strain & Fatigue
    tab. Called from `tab_fatigue()` for the initial render and from
    `update_fatigue_output()` whenever a slider or the scenario
    dropdown changes."""
    side = _tab_fatigue_sidebar(f)

    right_pane = html.Div([
        # Scenario-specific section (figures + tables + verdict)
        render_fatigue_for_scenario(scenario_name, f=f),

        # ALL-SCENARIO COMPARISON
        section_header('All scenarios at a glance'),
        dcc.Graph(figure=_fatigue_all_scenarios_figure(f=f),
                   config={'displayModeBar': False}),
        html.Div('Scenarios summary table', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '20px 0 8px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        _fatigue_scenarios_summary_table(f=f),
    ])

    return html.Div([side, right_pane], style=STYLES['two_col'])


def _fatigue_scenarios_summary_table(f=None):
    """Styled summary table of all 6 scenarios."""
    if f is None:
        f = DATA['fatigue']
    s = f['scenarios']

    rows = []
    for name, r in s.items():
        years_str = (f'{r["years_to_fail"]:.1f} yr'
                      if r['d_per_yr'] > 0 else 'infinite')
        verdict_color = (COLORS['accent_2'] if r['overall_pass']
                          else COLORS['alert'])
        d_per_yr_str = ('0  (no damage)' if r['d_per_yr'] == 0
                         else fmt_value(r['d_per_yr']))
        rows.append([
            name,
            f'{r["max_stress"]:.1f} MPa',
            d_per_yr_str,
            years_str,
            html.Span('PASS' if r['overall_pass'] else 'FAIL',
                       style={'fontWeight': '700', 'color': verdict_color}),
        ])
    return styled_table(
        headers=['Scenario', 'Max σ', 'D / year',
                  'Years to D = 1', 'Verdict'],
        rows=rows,
    )


def _damage_vs_time_figure(r):
    """Plot damage D accumulating linearly with time.

    Marks the four standard milestones (1, 5, 10, 20 years) plus the
    failure-line crossing at D = 1.0. Pure function of `r` so it
    automatically updates when fatigue parameters or scenarios change.
    """
    d_per_yr = r['d_per_yr']
    # Years to D = 1
    if d_per_yr <= 0:
        # No damage; just show 0 to 30 years
        years_plot = np.linspace(0, 30, 200)
        D_plot = d_per_yr * years_plot
        years_fail = None
    else:
        years_fail = 1.0 / d_per_yr
        # Plot range: 0 → max(25 yr, slightly past failure crossing)
        x_max = max(25.0, min(years_fail * 1.05, 1e6))
        years_plot = np.linspace(0, x_max, 400)
        D_plot = d_per_yr * years_plot

    fig = go.Figure()
    # Damage line
    fig.add_trace(go.Scatter(
        x=years_plot, y=D_plot,
        mode='lines',
        line=dict(color=COLORS['accent'], width=2.4),
        name='D(t)',
        hovertemplate='Year %{x:.1f}<br>D = %{y:.6f}<extra></extra>',
    ))

    # Marker dots at 1, 5, 10, 20 years
    milestone_years = [1, 5, 10, 20]
    milestone_D = [d_per_yr * y for y in milestone_years]
    fig.add_trace(go.Scatter(
        x=milestone_years, y=milestone_D,
        mode='markers+text',
        marker=dict(color=COLORS['accent'], size=9,
                     line=dict(color='white', width=1.5)),
        text=[f'{y} yr<br>D = {fmt_value(dy)}'
              for y, dy in zip(milestone_years, milestone_D)],
        textposition='top center',
        textfont=dict(size=10, color=COLORS['text'], family=FONT),
        showlegend=False,
        hoverinfo='skip',
    ))

    # D = 1.0 failure line
    fig.add_hline(y=1.0, line=dict(color=COLORS['alert'], width=1.5,
                                     dash='dash'))
    fig.add_annotation(
        x=years_plot[-1], y=1.0, xanchor='right', yanchor='bottom',
        text='D = 1.0  (failure threshold)',
        showarrow=False,
        font=dict(size=11, color=COLORS['alert'], family=FONT,
                   weight='bold'),
        yshift=4,
    )

    # Mark the failure-line crossing if it's within range
    if years_fail is not None and years_fail <= years_plot[-1]:
        fig.add_trace(go.Scatter(
            x=[years_fail], y=[1.0],
            mode='markers+text',
            marker=dict(color=COLORS['alert'], size=11, symbol='x',
                         line=dict(width=2)),
            text=[f'  D = 1 at {years_fail:.1f} yr'],
            textposition='middle right',
            textfont=dict(size=11, color=COLORS['alert'], family=FONT,
                           weight='bold'),
            showlegend=False,
            hoverinfo='skip',
        ))

    # y axis: log if D=1 is way above the milestone points
    # Otherwise linear so the dashed failure line is clearly visible.
    max_y_data = max(milestone_D[-1], 1.0) * 1.4
    fig.update_layout(plot_layout(
        title=dict(
            text=f'Damage accumulation  —  '
                  f'D(t) = D_per_year × t  '
                  f'(D_per_year = {fmt_value(d_per_yr)})',
            font=dict(size=12)),
        xaxis_title='Operational time (years)',
        yaxis_title='Cumulative damage  D',
        height=320,
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=45),
    ))
    fig.update_yaxes(range=[0, max_y_data])
    return fig


def _fatigue_all_scenarios_figure(f=None):
    """Bar chart of annual damage for all 6 scenarios."""
    if f is None:
        f = DATA['fatigue']
    s = f['scenarios']
    names = list(s.keys())
    d_vals = [s[n]['d_per_yr'] for n in names]
    # For log scale, replace zero with tiny value for plotting (annotate true value)
    d_plot = [max(v, 1e-6) for v in d_vals]

    def _fmt(v):
        """Decimal-friendly formatting (avoid scientific notation)."""
        if v == 0:
            return '0  (no damage)'
        if v >= 1:
            return f'{v:.1f}'
        if v >= 0.01:
            return f'{v:.3f}'
        if v >= 0.0001:
            return f'{v:.5f}'
        return f'{v:.7f}'

    text = [_fmt(v) for v in d_vals]
    fig = go.Figure(go.Bar(
        x=names, y=d_plot,
        marker_color=COLORS['text'],         # black bars
        marker_line=dict(color='black', width=0.5),
        text=text,
        textposition='outside',
        textfont=dict(size=11),
        width=0.35,                          # narrower bars (default ≈0.8)
    ))
    fig.add_hline(y=1.0, line_dash='dash', line_color=COLORS['alert'],
                  annotation_text='D / year = 1 (fails in 1 year)',
                  annotation_position='top right')
    fig.update_layout(plot_layout(
        yaxis_type='log',
        yaxis_title='Annual damage  D / year   (log scale)',
        yaxis=dict(range=[-6, 2.5]),
        title=dict(text='Annual fatigue damage by sea state',
                    font=dict(size=13)),
        height=380,
        showlegend=False,
        bargap=0.5,                          # extra spacing between bars
    ))
    return fig


def _sn_curve_figure(sigma_lim, N_lim, m, sigma_max=300.0):
    """Plot S-N curve on log-log axes for the chosen weld class + environment.

    N(σ) = N_lim · (σ_lim / σ)^m       for σ > σ_lim
    N(σ) = ∞                            for σ ≤ σ_lim  (drawn as horizontal line)
    """
    sigma_range = np.linspace(sigma_lim*1.01, sigma_max, 200)
    N_range = N_lim * (sigma_lim / sigma_range) ** m

    fig = go.Figure()
    # S-N curve above fatigue limit
    fig.add_trace(go.Scatter(
        x=N_range, y=sigma_range,
        line=dict(color=COLORS['accent'], width=2.5),
        name='S-N curve  N = N_lim · (σ_lim/σ)^m',
    ))
    # Horizontal line at fatigue limit (below it, N is infinite)
    fig.add_hline(y=sigma_lim, line_dash='dash', line_color=COLORS['accent_2'],
                   annotation_text=f'fatigue limit σ_lim = {sigma_lim:.0f} MPa  '
                                    f'(below: infinite life)',
                   annotation_position='bottom right')
    # Mark the (N_lim, σ_lim) anchor
    fig.add_trace(go.Scatter(
        x=[N_lim], y=[sigma_lim], mode='markers',
        marker=dict(size=12, color=COLORS['accent_2'], symbol='circle',
                     line=dict(color='white', width=2)),
        name=f'anchor: ({fmt_value(N_lim)} cycles, {sigma_lim:.0f} MPa)',
    ))
    fig.update_layout(plot_layout(
        title=dict(text='S-N curve — BS 7608 Class D + seawater factor 0.75',
                    font=dict(size=13)),
        height=380,
        legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center',
                     font=dict(size=10)),
    ))
    # Explicit log-axis ranges & ticks (avoid Plotly auto-scaling to 10^20)
    fig.update_xaxes(
        type='log',
        title_text='N — cycles to failure  (log scale)',
        range=[np.log10(1e4), np.log10(2e7)],
        tickvals=[1e4, 1e5, 1e6, 1e7],
        ticktext=['10⁴', '10⁵', '10⁶', '10⁷'],
    )
    fig.update_yaxes(
        type='log',
        title_text='σ — cycle stress range (MPa, log scale)',
        range=[np.log10(40), np.log10(400)],
        tickvals=[40, 60, 80, 100, 200, 300, 400],
        ticktext=['40', '60', '80', '100', '200', '300', '400'],
    )
    return fig


def render_fatigue_for_scenario(scenario_name, f=None):
    """Build the scenario-specific section (called by callback).
    Comprehensive walk-through: scenario mapping → signal model → Hooke →
    filter → rainflow → BS 7608 reference → S-N curve → Miner → lifetime →
    hard limits → final verdict.

    `f` is the fatigue data dict (default: global DATA['fatigue']). Pass an
    in-memory dict here to render with user-tuned material parameters.
    """
    if f is None:
        f = DATA['fatigue']
    r = f['scenarios'][scenario_name]
    p = r['params']
    mat = f['material']
    fp = f['filter_params']
    bs = f['bs7608_classes']
    envs = f['env_factors']
    cls_used = f['weld_class_used']
    env_used = f['environment']

    # ============== FIG 1: strain & stress time series ==============
    fig1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12,
        subplot_titles=(
            'Gauge reading — strain ε(t)  (in microstrain)',
            'After Hooke conversion — stress σ(t) = E · ε  (in MPa)',
        ),
    )
    fig1.add_trace(go.Scatter(x=r['t']/60, y=r['strain_us'],
                               line=dict(color=COLORS['accent_2'], width=0.6),
                               showlegend=False), row=1, col=1)
    fig1.add_trace(go.Scatter(x=r['t']/60, y=r['stress_raw'],
                               line=dict(color=COLORS['accent'], width=0.6),
                               showlegend=False), row=2, col=1)
    fig1.update_layout(plot_layout(height=380))
    fig1.update_xaxes(title_text='Time (min)', row=2, col=1)
    fig1.update_yaxes(title_text='ε (με)', row=1, col=1)
    fig1.update_yaxes(title_text='σ (MPa)', row=2, col=1)
    for ann in fig1['layout']['annotations']:
        ann['font'] = dict(size=12, color=COLORS['text'])

    # ============== FIG 2: raw vs filtered stress (first 60 s) ==============
    mask = r['t'] < 60
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=r['t'][mask], y=r['stress_raw'][mask],
        name='Raw stress  (noise + waves)',
        line=dict(color=COLORS['warning'], width=0.7), opacity=0.7))
    fig2.add_trace(go.Scatter(
        x=r['t'][mask], y=r['stress_filt'][mask],
        name=f'Low-pass filtered  (cutoff {fp["cutoff"]} Hz)',
        line=dict(color=COLORS['accent'], width=1.5)))
    fig2.update_layout(plot_layout(
        title=dict(text='First 60 s — Butterworth low-pass removes sensor noise, '
                         'preserves the wave content at 0.125 Hz',
                    font=dict(size=13)),
        xaxis_title='Time (s)', yaxis_title='σ (MPa)',
        height=300,
        legend=dict(orientation='h', y=-0.18, x=0.5, xanchor='center',
                     font=dict(size=10)),
    ))

    # ============== FIG 3: rainflow histogram (binned) ==============
    # Rainflow returns (range, count) pairs where count is typically 0.5 or 1.0.
    # Plotting them directly creates a near-flat bar at y∈[0.5, 1] — useless.
    # Instead: bin the ranges (weighted by counts) into a proper histogram.
    ranges = [s for s, _ in r['cycles_filt']]
    counts = [c for _, c in r['cycles_filt']]
    if ranges:
        x_max = max(ranges) * 1.05
        bins = np.linspace(0, x_max, 30)
        hist_y, _ = np.histogram(ranges, bins=bins, weights=counts)
        bin_centers = (bins[:-1] + bins[1:]) / 2
    else:
        bin_centers, hist_y = [], []
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        x=bin_centers, y=hist_y,
        marker_color=COLORS['accent'],
        marker_line=dict(color='white', width=0.5),
        showlegend=False,
    ))
    fig3.add_vline(x=mat['sigma_limit'], line_dash='dash', line_color=COLORS['alert'],
                    annotation_text=f"σ_lim = {mat['sigma_limit']:.0f} MPa  "
                                     f"(below: no damage)",
                    annotation_position='top right')
    fig3.update_layout(plot_layout(
        title=dict(text='Rainflow cycle-range distribution — only bars right of '
                         'the dashed line damage the weld',
                    font=dict(size=13)),
        xaxis_title='Cycle stress range (MPa)',
        yaxis_title='Cycle count per bin',
        height=320,
        showlegend=False,
        bargap=0.02,
    ))

    # ============== FIG 4: S-N curve ==============
    fig_sn = _sn_curve_figure(mat['sigma_limit'], mat['N_limit'], mat['m_sn'])

    # ============== FIG 5: hard-limits bar (3 bars only) ==============
    # Show only what we actually compare: peak observed vs the two allowables.
    # Raw yield/UTS spec (250/450) lives in the Step 9 table, not the chart.
    # Monte Carlo histogram of σ_peak: distribution across N independent
    # 30-min realizations of the scenario. The histogram + the two
    # allowable thresholds tell us *how often* an exceedance occurs.
    fig5 = go.Figure()
    fig5.add_trace(go.Histogram(
        x=r['mc_peaks'],
        nbinsx=24,
        marker_color=COLORS['accent'],
        marker_line=dict(color='white', width=0.5),
        opacity=0.85,
        name='σ_peak (Monte Carlo)',
    ))
    # Vertical lines for the two allowables
    yield_allow = mat['yield_str'] / mat['yield_sf']
    uts_allow   = mat['UTS']       / mat['UTS_sf']
    for x_val, label, col, y_paper in [
        (yield_allow, f'Yield / SF = {yield_allow:.0f} MPa',  COLORS['warning'], 0.96),
        (uts_allow,   f'UTS / SF = {uts_allow:.0f} MPa',     COLORS['alert'],   0.84),
    ]:
        fig5.add_vline(x=x_val, line=dict(color=col, width=1.6, dash='dash'))
        fig5.add_annotation(
            x=x_val, y=y_paper, xref='x', yref='paper',
            text=label, showarrow=False,
            font=dict(size=10, color=col, family=FONT, weight='bold'),
            bgcolor='rgba(255,255,255,0.85)',
            xanchor='left', yanchor='top', xshift=4,
        )
    # Mean line
    fig5.add_vline(x=r['mc_peak_mean'],
                    line=dict(color=COLORS['text'], width=1.4))
    fig5.add_annotation(
        x=r['mc_peak_mean'], y=0.92, xref='x', yref='paper',
        text=f"mean = {r['mc_peak_mean']:.1f} MPa",
        showarrow=False, font=dict(size=10, color=COLORS['text'],
                                       family=FONT, weight='bold'),
        xanchor='right', xshift=-4,
    )
    fig5.update_layout(plot_layout(
        title=dict(text=f"σ_peak distribution from Monte Carlo  "
                          f"({r['mc_n_runs']} independent 30-min realisations)",
                    font=dict(size=12)),
        xaxis_title='Peak observed stress  σ_peak  (MPa)',
        yaxis_title='Number of realisations',
        height=300,
        showlegend=False,
        bargap=0.05,
    ))

    # Helper for highlighting the current row in tables
    def highlight_if(condition, base_text):
        if condition:
            return html.Span(base_text, style={
                'fontWeight': '700', 'color': COLORS['accent_2'],
            })
        return base_text

    years_str = f"{r['years_to_fail']:.1f} yr" if r['d_per_yr'] > 0 else "infinite"
    pct_life_30min = r['d_30min'] * 100

    # ============== ASSEMBLE EVERYTHING ==============
    return html.Div([

        # ===== Section 0: scenario mapping =====
        section_header(f'Sea-state scenario — currently: {scenario_name}'),
        dcc.Markdown(
            "Each scenario corresponds to a real sea state. The strain "
            "gauge reads more or less depending on wave loading on the "
            "neck weld — and that signal is the start of the fatigue "
            "analysis.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        styled_table(
            headers=['Scenario', 'H_s (m)', 'T_p (s)', 'Beaufort',
                      'Strain ε_amp', 'Description'],
            rows=[
                [highlight_if(name == scenario_name, name),
                  highlight_if(name == scenario_name,
                                f"{f['scenarios'][name]['params']['Hs_m']:.1f}"),
                  highlight_if(name == scenario_name,
                                f"{f['scenarios'][name]['params']['Tp_s']:.0f}"),
                  highlight_if(name == scenario_name,
                                f['scenarios'][name]['params']['beaufort']),
                  highlight_if(name == scenario_name,
                                f"{f['scenarios'][name]['params']['strain_amplitude']} με"),
                  highlight_if(name == scenario_name,
                                f['scenarios'][name]['params']['description'])]
                for name in f['scenarios'].keys()
            ],
        ),

        # ============================================
        # ===== PART 1: cumulative fatigue damage
        # ============================================
        part_header('PART 1 — Cumulative fatigue damage  (many cycles, slow degradation)'),
        dcc.Markdown(
            "Fatigue is **damage accumulated over millions of cycles**. "
            "Each cycle eats a tiny fraction of the weld's life; if "
            "enough cycles accumulate, the weld eventually cracks. "
            "Steps 1-8 walk through this calculation end-to-end.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'marginBottom': '8px'},
        ),

        # ===== Section 1: signal model =====
        section_header('Step 1 — the strain signal model'),
        dcc.Markdown(
            "The strain gauge bonded to the critical weld reads a "
            "time-varying signal. We model it as a wave-period carrier, "
            "modulated by a longer wave-group envelope, plus a constant "
            "mean offset and broadband sensor noise:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'\varepsilon(t) \;=\; \underbrace{\varepsilon_{\text{amp}}\sin\!\left(\frac{2\pi t}{T_{\text{wave}}}\right)}_{\text{carrier}} \,\cdot\, \underbrace{\left(1 + \delta_{\text{env}}\sin\!\left(\frac{2\pi t}{T_{\text{env}}}\right)\right)}_{\text{wave-group envelope}} \;+\; \varepsilon_{\text{mean}} \;+\; n(t)'),
        styled_table(
            headers=['Parameter', 'Symbol', 'Value (this scenario)'],
            rows=[
                ['Strain amplitude',       'ε_amp',        f'{p["strain_amplitude"]} microstrain'],
                ['Mean strain (offset)',   'ε_mean',       '200 microstrain'],
                ['Envelope depth',         'δ_env',        f'{p["envelope_depth"]:.2f}'],
                ['Wave period',            'T_wave',       '8.0 s  (carrier)'],
                ['Wave-group period',      'T_env',        '300 s  (5-min envelope)'],
                ['Noise std',              'σ_n',          f'{p["noise_level"]} microstrain RMS'],
                ['Sample rate',            'f_s',          f'{fp["fs"]} Hz'],
                ['Record duration',        'T_record',     '30 minutes'],
            ],
        ),
        dcc.Graph(figure=fig1, config={'displayModeBar': False}),

        # ===== Section 2: Hooke's law =====
        section_header("Step 2 — strain → stress via Hooke's law"),
        dcc.Markdown(
            "BS 7608 (and every other fatigue standard) works in **stress**, "
            "not strain. So we convert with Hooke's law:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'\sigma \;=\; E \cdot \varepsilon'),
        dcc.Markdown(
            f"For structural steel $E = {mat['E']:,.0f}$ MPa. So if "
            f"$\\varepsilon$ is expressed in microstrain "
            f"$(10^{{-6}})$:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'\sigma_{\text{MPa}} \;=\; \varepsilon_{\mu\varepsilon} \cdot 0.2 \qquad \text{(rule of thumb: 1 microstrain ≈ 0.2 MPa for steel)}'),
        styled_table(
            headers=['Quantity', 'Value (this scenario)'],
            rows=[
                ['Carrier strain amplitude',  f'{p["strain_amplitude"]} microstrain'],
                ['Carrier stress amplitude',  f'{p["strain_amplitude"] * 0.2:.1f} MPa'],
                ["Young's modulus E",         f'{mat["E"]:,.0f} MPa  (= 200 GPa)'],
            ],
        ),

        # ===== Section 3: low-pass filter =====
        section_header('Step 3 — pre-processing (low-pass filter)'),
        dcc.Markdown(
            "Sensor noise creates spurious turning points, which a "
            "downstream cycle-counter would treat as real cycles. The fix: "
            "filter out everything above the wave frequency. A Butterworth "
            f"low-pass with cutoff $f_c = {fp['cutoff']}$ Hz preserves "
            f"wave content ($f_{{\\text{{wave}}}} = {1/8:.3f}$ Hz) and "
            f"removes the rest.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        styled_table(
            headers=['Filter parameter', 'Value'],
            rows=[
                ['Type',                       'Butterworth low-pass'],
                ['Order',                      f'{fp["order"]}'],
                ['Cutoff frequency f_c',       f'{fp["cutoff"]} Hz'],
                ['Wave content (preserved)',   f'{1/8:.3f} Hz  (T = 8 s)'],
                ['Implementation',             'scipy.signal.filtfilt  (zero phase shift)'],
            ],
        ),
        dcc.Graph(figure=fig2, config={'displayModeBar': False}),

        # ===== Section 4: rainflow =====
        section_header('Step 4 — rainflow cycle counting'),
        dcc.Markdown(
            "**Rainflow counting** (ASTM E1049, industry standard since the "
            "1960s) extracts individual loading cycles from a time series. "
            "Output: a list of `(stress_range, count)` pairs — every cycle "
            "the structure experienced, regardless of size. We feed it the "
            "filtered stress signal from Step 3.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        styled_table(
            headers=['Quantity', 'Value (this scenario)'],
            rows=[
                ['Raw cycles (noisy signal)',
                  html.Span(f'{r["n_raw"]}',
                             style={'color': COLORS['text_dim']})],
                ['Filtered cycles (after low-pass)',
                  html.Span(f'{r["n_filt"]}',
                             style={'fontWeight': '700',
                                     'color': COLORS['accent']})],
                ['Max cycle range observed', f'{r["max_cycle_range"]:.1f} MPa'],
            ],
            footer_note='The big drop from raw to filtered cycles is the noise — '
                        'rainflow on a noisy signal massively overcounts. '
                        'Filtered count is what we use for damage analysis.',
        ),
        dcc.Graph(figure=fig3, config={'displayModeBar': False}),
        dcc.Markdown(
            f"The dashed line at $\\sigma_{{\\text{{lim}}}} = "
            f"{mat['sigma_limit']:.0f}$ MPa is shown for context — it comes "
            f"from BS 7608 (Step 5). Rainflow itself doesn't know about "
            f"$\\sigma_{{\\text{{lim}}}}$; it just enumerates every cycle. "
            f"The above-vs-below classification happens later, in Step 7.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'marginTop': '8px'},
        ),

        # ===== Section 5: BS 7608 reference table =====
        section_header('Step 5 — BS 7608 weld class & seawater factor'),
        dcc.Markdown(
            "**BS 7608** is the British Standard for fatigue design of "
            "welded steel structures (offshore industry default). It "
            "specifies a small set of weld classes, each with its own "
            "fatigue limit. Worse welds → lower limit → fewer cycles to "
            "failure at any given stress.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        styled_table(
            headers=['Class', 'σ_lim, air (MPa)', 'Description'],
            rows=[
                [highlight_if(cls == cls_used, f'Class {cls}'),
                  highlight_if(cls == cls_used, f"{bs[cls]['sigma_limit_air']:.0f}"),
                  highlight_if(cls == cls_used, bs[cls]['description'])]
                for cls in ['B', 'D', 'F', 'W']
            ],
            footer_note='All classes use the same S-N slope m = 3 and the '
                        'same anchor N_limit = 1e7 cycles — what differs '
                        'between classes is the fatigue-limit stress.',
        ),
        dcc.Markdown(
            "Seawater accelerates crack growth, so standards apply a "
            "reduction factor to the in-air fatigue limit:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65', 'margin': '12px 0 6px 0'},
        ),
        styled_table(
            headers=['Environment', 'Factor', 'When to use'],
            rows=[
                [highlight_if(env == env_used, env),
                  highlight_if(env == env_used, f'× {envs[env]:.2f}'),
                  highlight_if(env == env_used,
                                {'air': 'Above-water structure',
                                 'seawater_cathodic': 'Submerged, with cathodic protection (typical offshore)',
                                 'seawater_free_corrosion': 'Submerged, unprotected'}[env])]
                for env in ['air', 'seawater_cathodic', 'seawater_free_corrosion']
            ],
        ),
        latex_equation(
            r'\sigma_{\text{lim, eff}} \;=\; \sigma_{\text{lim, air}} \cdot f_{\text{env}} '
            rf'\;=\; {mat["sigma_limit_air"]:.0f} \cdot {mat["env_factor"]:.2f} \;=\; {mat["sigma_limit"]:.0f} \;\;\text{{MPa}}'
        ),

        # === Classification of Step 4's cycles ===
        dcc.Markdown(
            f"Now that we have $\\sigma_{{\\text{{lim}}}}$, we can split "
            f"the Step 4 rainflow cycles into damaging vs non-damaging "
            f"groups — cycles **above** the fatigue limit consume life, "
            f"cycles **below** it don't (the S-N curve's infinite-life "
            f"branch).",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '14px 0 6px 0'},
        ),
        styled_table(
            headers=['Quantity', 'Value (this scenario)'],
            rows=[
                ['Filtered cycles total  (from Step 4)',
                  f'{r["n_filt"]}'],
                ['Cycles above σ_lim  (damaging)',
                  html.Span(f'{r["n_damaging"]}',
                             style={'fontWeight': '700',
                                     'color': COLORS['alert']
                                              if r['n_damaging'] > 0
                                              else COLORS['accent_2']})],
                ['Cycles below σ_lim  (no damage)',
                  f'{r["n_safe"]}'],
            ],
        ),

        # ===== Section 6: S-N curve — minimal, just the bridge =====
        section_header('Step 6 — the S-N curve'),
        dcc.Markdown(
            "We have the damaging cycles. We still need the bridge from "
            "**stress range $\\sigma$** to **cycles-to-failure "
            "$N(\\sigma)$**. That bridge is the S-N curve — a power-law "
            "model from BS 7608, anchored at the fatigue limit:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'N(\sigma) \;=\; N_{\text{lim}} \cdot \left(\dfrac{\sigma_{\text{lim}}}{\sigma}\right)^{m} \qquad (m = 3)'),
        dcc.Graph(figure=fig_sn, config={'displayModeBar': False}),
        dcc.Markdown(
            f"**How we use it:** for each damaging cycle from Step 5 at "
            f"stress range $\\sigma_i$, read off $N(\\sigma_i)$ from this "
            f"curve. That one cycle consumes the fraction "
            f"$1/N(\\sigma_i)$ of the weld's fatigue life. Step 7 sums "
            f"these fractions over all cycles.\n\n"
            f"*(On log-log axes a power law plots as a straight line — "
            f"this is just algebra, not an empirical fit. Below "
            f"$\\sigma_{{\\text{{lim}}}}$ the curve runs flat to "
            f"infinity, the no-damage region.)*",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'marginTop': '8px'},
        ),

        # ===== Section 7: Miner's rule + damage calculation =====
        section_header("Step 7 — Miner's rule, damage accumulation"),
        dcc.Markdown(
            f"**What does $N_{{\\text{{lim}}}}$ mean physically?** "
            f"$N_{{\\text{{lim}}}} = {fmt_value(mat['N_limit'])}$ is the "
            f"*number of cycles at exactly the fatigue limit* "
            f"$\\sigma_{{\\text{{lim}}}}$ that would use up 100% of the "
            f"weld's fatigue life. So one cycle at "
            f"$\\sigma_{{\\text{{lim}}}}$ eats "
            f"$1/N_{{\\text{{lim}}}} = {fmt_value(1/mat['N_limit'])}$ of "
            f"the weld's life. Other stress levels follow the S-N curve.\n\n"
            f"**Miner's rule** says damage from cycles of different "
            f"stress sizes accumulates **linearly** — small cycles eat a "
            f"tiny bit, large cycles eat a lot, but they all eat from "
            f"the same finite reservoir. So we just sum cycle-by-cycle "
            f"damage to get the total.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        html.Div('Step 7a — damage per cycle', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '16px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        latex_equation(r'd(\sigma) \;=\; \dfrac{1}{N(\sigma)} \;=\; \dfrac{1}{N_{\text{lim}}} \cdot \left(\dfrac{\sigma}{\sigma_{\text{lim}}}\right)^{m}'),
        dcc.Markdown(
            f"With $m = 3$ this is a cube law — **doubling the stress "
            f"makes one cycle 8× more damaging**. This is the single "
            f"most important fact in fatigue design.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        html.Div("Step 7b — total damage in the record", style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '20px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        latex_equation(r'D \;=\; \sum_{i} n_{i} \cdot d(\sigma_{i}) \;=\; \sum_{i} \dfrac{n_{i}}{N(\sigma_{i})}'),
        dcc.Markdown(
            "where $n_i$ is the count of rainflow cycles at stress "
            "range $\\sigma_i$, summed over the Step 4 histogram. "
            "**Failure is predicted at $D = 1.0$** — the weld has used "
            "100% of its fatigue capacity.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ===== Step 8: damage accumulation plot + bullets =====
        section_header(f'Step 8 — damage accumulation for the {scenario_name} scenario'),
        dcc.Markdown(
            "Damage accumulates **linearly** with time (Miner's rule). "
            "The dashed line marks failure at $D = 1.0$:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Graph(figure=_damage_vs_time_figure(r),
                   config={'displayModeBar': False}),
        dcc.Markdown(
            f"- **Damage per year** is "
            f"$D_{{\\text{{year}}}} = {fmt_value(r['d_per_yr'])}$.\n"
            f"- **Damage at 20-year design life** is "
            f"$D_{{20\\,\\text{{yr}}}} = {fmt_value(r['d_20yr'])}$ "
            f"— {'**below 1.0, passes**' if r['fatigue_pass'] else '**above 1.0, FAILS**'}.\n"
            f"- $D$ reaches 1.0 at **{years_str}**.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ============================================
        # ===== PART 2: hard limits
        # ============================================
        part_header('PART 2 — Hard limits  (single events, one is enough)'),
        dcc.Markdown(
            "Fatigue is about **many cycles, slow degradation**. There are "
            "also **static** failure modes — a single overload event is "
            "enough to damage the structure. Two static limits matter for "
            "steel:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Markdown(
            "- **Yield strength  $\\sigma_y$** — exceeding it causes "
            "**permanent plastic deformation**: the steel does not "
            "return to its original shape after unloading.\n"
            "- **Ultimate tensile strength  $\\sigma_{\\text{UTS}}$** — "
            "exceeding it causes **fracture**: the structure breaks apart.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        section_header('Step 9 — material spec & allowable peaks'),
        dcc.Markdown(
            "Industry practice applies a **safety factor** on top of "
            "each spec limit. The right column below is the **allowable "
            "peak stress** — what we'll compare the observed peak "
            "against in Step 10. The middle column is the raw material "
            "spec, shown for context.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        styled_table(
            headers=['Limit', 'Steel value', 'Safety factor', 'Allowable peak stress'],
            rows=[
                ['Yield strength (S235 steel)',
                  f'{mat["yield_str"]:.0f} MPa',
                  f'{mat["yield_sf"]:.1f}',
                  html.Span(f'{mat["yield_str"]/mat["yield_sf"]:.0f} MPa',
                             style={'fontWeight': '700',
                                     'color': COLORS['warning']})],
                ['Ultimate tensile strength',
                  f'{mat["UTS"]:.0f} MPa',
                  f'{mat["UTS_sf"]:.1f}',
                  html.Span(f'{mat["UTS"]/mat["UTS_sf"]:.0f} MPa',
                             style={'fontWeight': '700',
                                     'color': COLORS['warning']})],
            ],
        ),

        section_header('Step 10 — peak observed stress vs allowable'),
        dcc.Markdown(
            "The hard-limit check uses the **peak observed stress** — the "
            "largest excursion anywhere in the **raw** stress signal "
            "(Step 2 output, before the Step 3 low-pass filter). We use "
            "the raw signal because real overload events are short, "
            "high-frequency, and would be smoothed out by the filter.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'\sigma_{\text{peak}} \;=\; \max_{t}\,\big|\,\sigma(t) - \overline{\sigma}\,\big|'),
        dcc.Markdown(
            f"**Monte Carlo over the simulation window.** A single "
            f"30-minute realisation gives one $\\sigma_{{\\text{{peak}}}}$ "
            f"value — but every 30-minute window at sea is statistically "
            f"different. We run **{r['mc_n_runs']} independent "
            f"realisations** of the {scenario_name} scenario (same wave "
            f"amplitude and envelope; independent noise + phase via "
            f"different RNG seeds) and collect "
            f"$\\sigma_{{\\text{{peak}}}}$ from each.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Graph(figure=fig5, config={'displayModeBar': False}),
        styled_table(
            headers=['Statistic', f'σ_peak  (MPa) over {r["mc_n_runs"]} runs',
                      'vs Yield / SF', 'vs UTS / SF'],
            rows=[
                ['Mean',
                  f'{r["mc_peak_mean"]:.1f}',
                  '✓ below' if r['mc_peak_mean'] < mat['yield_str']/mat['yield_sf'] else 'ABOVE',
                  '✓ below' if r['mc_peak_mean'] < mat['UTS']/mat['UTS_sf'] else 'ABOVE'],
                ['95th percentile',
                  f'{r["mc_peak_p95"]:.1f}',
                  '✓ below' if r['mc_peak_p95'] < mat['yield_str']/mat['yield_sf'] else 'ABOVE',
                  '✓ below' if r['mc_peak_p95'] < mat['UTS']/mat['UTS_sf'] else 'ABOVE'],
                ['Worst case  (max)',
                  html.Span(f'{r["mc_peak_max"]:.1f}',
                             style={'fontWeight': '700',
                                     'color': COLORS['accent']}),
                  html.Span(
                      '✓ below' if r['mc_peak_max'] < mat['yield_str']/mat['yield_sf'] else 'ABOVE',
                      style={'color': COLORS['accent_2']
                              if r['mc_peak_max'] < mat['yield_str']/mat['yield_sf']
                              else COLORS['alert'], 'fontWeight': '700'}),
                  html.Span(
                      '✓ below' if r['mc_peak_max'] < mat['UTS']/mat['UTS_sf'] else 'ABOVE',
                      style={'color': COLORS['accent_2']
                              if r['mc_peak_max'] < mat['UTS']/mat['UTS_sf']
                              else COLORS['alert'], 'fontWeight': '700'}),
                 ],
            ],
            footer_note=f'A real fleet-monitoring system would re-evaluate '
                        f'σ_peak continuously on the actual gauge signal, '
                        f'in rolling 30-min or 10-min windows. The Monte '
                        f'Carlo here gives us the distribution to expect '
                        f'in a single window.',
        ),
        styled_table(
            headers=['Check', 'σ_peak (worst case)',
                      'Allowable', 'Margin', 'Verdict'],
            rows=[
                ['Yield  (σ_peak < σ_y / SF)',
                  f'{r["mc_peak_max"]:.1f} MPa',
                  f'{mat["yield_str"]/mat["yield_sf"]:.0f} MPa',
                  html.Span(
                      f'{mat["yield_str"]/mat["yield_sf"]/r["mc_peak_max"]:.1f}×' if r['mc_peak_max'] > 0 else '∞',
                      style={'fontWeight': '700',
                              'color': COLORS['accent_2']
                                       if r['mc_peak_max'] < mat['yield_str']/mat['yield_sf']
                                       else COLORS['alert']}),
                  html.Span(
                      'PASS' if r['mc_peak_max'] < mat['yield_str']/mat['yield_sf'] else 'FAIL',
                      style={'fontWeight': '700',
                              'color': COLORS['accent_2']
                                       if r['mc_peak_max'] < mat['yield_str']/mat['yield_sf']
                                       else COLORS['alert']})],
                ['Fracture  (σ_peak < σ_UTS / SF)',
                  f'{r["mc_peak_max"]:.1f} MPa',
                  f'{mat["UTS"]/mat["UTS_sf"]:.0f} MPa',
                  html.Span(
                      f'{mat["UTS"]/mat["UTS_sf"]/r["mc_peak_max"]:.1f}×' if r['mc_peak_max'] > 0 else '∞',
                      style={'fontWeight': '700',
                              'color': COLORS['accent_2']
                                       if r['mc_peak_max'] < mat['UTS']/mat['UTS_sf']
                                       else COLORS['alert']}),
                  html.Span(
                      'PASS' if r['mc_peak_max'] < mat['UTS']/mat['UTS_sf'] else 'FAIL',
                      style={'fontWeight': '700',
                              'color': COLORS['accent_2']
                                       if r['mc_peak_max'] < mat['UTS']/mat['UTS_sf']
                                       else COLORS['alert']})],
            ],
        ),

        # ============================================
        # ===== FINAL: three-criteria verdict
        # ============================================
        part_header('Final design verdict — three criteria'),
        verdict_badge(
            'Criterion 1 — Fatigue (D < 1.0 over 20 years)',
            r['fatigue_pass'],
            f"D_20yr = {fmt_value(r['d_20yr'])}"),
        verdict_badge(
            f"Criterion 2 — Yield (peak < {mat['yield_str']:.0f}/{mat['yield_sf']} MPa)",
            r['yield_pass'],
            f"peak (worst case) = {r['mc_peak_max']:.1f} MPa"),
        verdict_badge(
            f"Criterion 3 — UTS (peak < {mat['UTS']:.0f}/{mat['UTS_sf']} MPa)",
            r['uts_pass'],
            f"peak (worst case) = {r['mc_peak_max']:.1f} MPa"),
        overall_verdict(r['overall_pass']),
    ])


def tab_strain():
    """Tab 3: Resonance tab — wrapper that builds the slider UI and a
    dynamic container that gets updated by the slider callback. The
    parameter-dependent content (the equations, tables, plots) is built
    by `_tab_strain_body(d)`.
    """
    d_init = DATA['strain']
    p_init = d_init['params']

    # ----- Slider definitions (all values are integers in display units) -----
    slider_box_style = {
        'background': '#F5F5F0',
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '4px',
        'padding': '14px 18px',
        'margin': '0 0 20px 0',
    }
    slider_label_style = {
        'fontSize': '12px', 'fontWeight': '600',
        'color': COLORS['text_dim'],
        'textTransform': 'uppercase',
        'letterSpacing': '0.04em',
        'margin': '0 0 6px 0',
    }

    def _slider_block(label_html, slider):
        return html.Div([
            html.Div(label_html, style=slider_label_style),
            slider,
        ], style={'flex': '1 1 0', 'minWidth': '160px'})

    slider_zeta = dcc.Slider(
        id='res-zeta', min=0.02, max=0.30, step=0.02,
        value=p_init['zeta'],
        marks={v/100: f'{v/100:.2f}' for v in [2, 5, 10, 20, 30]},
        tooltip={'placement': 'bottom', 'always_visible': False},
    )
    slider_F = dcc.Slider(
        id='res-Fdemo-kN', min=1, max=20, step=2,
        value=int(p_init['F_demo_N']/1000),
        marks={v: f'{v}' for v in [1, 5, 10, 15, 20]},
        tooltip={'placement': 'bottom', 'always_visible': False},
    )
    slider_m = dcc.Slider(
        id='res-m-tons', min=50, max=500, step=50,
        value=int(p_init['m']/1000),
        marks={v: f'{v}' for v in [50, 200, 350, 500]},
        tooltip={'placement': 'bottom', 'always_visible': False},
    )
    slider_k = dcc.Slider(
        id='res-k-kNpm', min=10, max=200, step=20,
        value=int(p_init['k']/1000),
        marks={v: f'{v}' for v in [10, 50, 100, 150, 200]},
        tooltip={'placement': 'bottom', 'always_visible': False},
    )

    sliders_panel = html.Div([
        html.Div('Tune the design — every number below updates as you '
                  'move a slider', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text'],
            'margin': '0 0 12px 0',
        }),
        html.Div([
            _slider_block('Damping ratio  ζ', slider_zeta),
            _slider_block('Effective mass  m  (tons)', slider_m),
            _slider_block('Stiffness  k  (kN/m)', slider_k),
            _slider_block('Demo forcing  F  (kN)', slider_F),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}),
    ], style=slider_box_style)

    return html.Div([
        tab_header(
            'Resonance — natural frequency of the float',
            "- **Mass-spring-damper model** of the float's heave "
            "dynamics — the simplest reasonable abstraction.\n"
            "- **Natural frequency estimated two ways** — theoretically "
            "from the physical parameters, and empirically from the "
            "spectrum of the strain response — and the two are cross-"
            "checked.\n"
            "- **Resonance scenario sweep**: quantifies the accumulated "
            "fatigue damage and the hard-limit margins (yield, "
            "ultimate tensile strength) when waves drive the float "
            "near its natural period."
        ),
        sliders_panel,
        dcc.Loading(
            id='resonance-loading',
            type='circle',
            color=COLORS['accent'],
            children=html.Div(id='resonance-output',
                                children=_tab_strain_body(d_init)),
        ),
    ])


def _tab_strain_body(d):
    """Build the parameter-dependent content of the Resonance tab from a
    pre-computed strain-data dict. Called from `tab_strain()` for the
    initial render and from `update_resonance_output()` whenever a
    slider moves.
    """
    scs = d['scenarios']
    p = d['params']

    # ----- Derived display values (everything below traces back to p / d) -----
    #   If the user changes m, k, c, ζ in make_strain_data(), every number on
    #   this tab updates automatically.
    p_m     = p['m']
    p_k     = p['k']
    p_c     = p['c']
    p_zeta  = p['zeta']
    omega_n = d['omega_n']
    T_n     = d['T_n']
    Q       = 1.0 / (2.0 * p_zeta)
    H_static_um_per_N    = 1e6 / p_k                # |H(0)|   in μm/N
    H_res_um_per_N       = 1e6 / (p_c * omega_n)    # |H(ωn)| in μm/N
    # A_wp from k = ρ g A_wp (so the prose stays consistent if k changes)
    A_wp_est             = p_k / (1025.0 * 9.8)
    # Demo forcing for the |H| table examples and the resonance scenarios
    F_demo_N             = p['F_demo_N']
    F_demo_kN            = F_demo_N / 1000.0
    kappa_us_per_m       = p['strain_per_meter']   # μstrain per meter heave

    side = html.Div([
        html.Div([
            html.Div('What is resonance', style=STYLES['sidebar_h']),
            dcc.Markdown(
                f"Every mechanical structure has one or more **natural "
                f"frequencies** $\\omega_n$ — frequencies at which it "
                f"*wants* to oscillate if you give it a push and let go. "
                f"If the external forcing happens to oscillate at "
                f"$\\omega_n$, the response amplitude blows up by a "
                f"factor of $Q = 1/(2\\zeta)$, where $\\zeta$ is the "
                f"damping ratio (chosen as {p_zeta:.2f} for this "
                f"dashboard, typical for floating structures). Stress "
                f"amplitude scales linearly with response amplitude; "
                f"fatigue damage scales as the **cube** of stress; so "
                f"even modest amplification at resonance translates to "
                f"**much faster** fatigue damage.",
                mathjax=True,
                style=STYLES['sidebar_body'],
            ),
        ], style=STYLES['sidebar_section']),

        html.Div([
            html.Div('What this tab does', style=STYLES['sidebar_h']),
            dcc.Markdown(
                "1. Models the float as a single-degree-of-freedom "
                "(SDOF) mass-spring-damper.\n"
                "2. **Estimates $\\omega_n$ two ways** — theoretically "
                "from $\\sqrt{k/m}$, and empirically from sensor data "
                "via FFT of the strain response. The two estimates "
                "should agree.\n"
                "3. Sweeps the wave period across $T_n$ and **estimates "
                "the accumulated fatigue damage and hard-limit margins** "
                "in each case — to quantify how dangerous resonance "
                "really is.",
                mathjax=True,
                style=STYLES['sidebar_body'],
            ),
        ], style=STYLES['sidebar_section']),
    ], style=STYLES['sidebar'])

    # =========== Figure 1: Transfer function ===========
    fig_tf = go.Figure()
    fig_tf.add_trace(go.Scatter(
        x=d['omega_range'], y=d['H_curve'], name='|H(ω)|',
        line=dict(color=COLORS['plot_1'], width=2),
    ))
    fig_tf.add_vline(x=d['omega_n'], line_dash='dash', line_color=COLORS['text_dim'],
                    annotation_text=f"ωₙ={d['omega_n']:.2f}",
                    annotation_position='top right')
    for sc in scs:
        om = 2*np.pi/sc['T_wave']
        fig_tf.add_trace(go.Scatter(
            x=[om], y=[sc['H_val']], mode='markers', name=sc['label'],
            marker=dict(size=14, color=sc['color'], line=dict(color='black', width=1.5)),
        ))
    fig_tf.update_layout(plot_layout(
        title=dict(text='Transfer function — bell curve peaks at natural frequency',
                    font=dict(size=14)),
        xaxis_title='Forcing angular frequency ω (rad/s)',
        yaxis_title='|H(ω)| (μm displacement per N of force)',
        yaxis_type='log',
        height=340,
    ))

    # =========== Figure 2: Resonance demo - FORCING + RESPONSE for 3 scenarios ===========
    fig_res = make_subplots(
        rows=2, cols=3, shared_xaxes='columns',
        subplot_titles=(*[sc['label'] for sc in scs], '', '', ''),
        horizontal_spacing=0.06, vertical_spacing=0.16,
        row_heights=[0.5, 0.5],
    )
    for col, sc in enumerate(scs, start=1):
        mask = sc['t'] < 60
        # Row 1: wave force (kN)
        fig_res.add_trace(go.Scatter(
            x=sc['t'][mask], y=sc['F_t'][mask]/1000,
            line=dict(color=sc['color'], width=1.0),
            showlegend=False,
        ), row=1, col=col)
        # Row 2: strain response (μstrain)
        fig_res.add_trace(go.Scatter(
            x=sc['t'][mask], y=sc['strain_clean'][mask],
            line=dict(color=sc['color'], width=1.0),
            showlegend=False,
        ), row=2, col=col)
        # Reference static line on response row
        fig_res.add_hline(y=100, line_dash='dot', line_color=COLORS['text_dim'],
                          opacity=0.5, row=2, col=col)
        fig_res.add_hline(y=-100, line_dash='dot', line_color=COLORS['text_dim'],
                          opacity=0.5, row=2, col=col)

    fig_res.update_xaxes(title_text='Time (s)', row=2)
    fig_res.update_yaxes(title_text='Force (kN)', row=1, col=1, range=[-6, 6])
    fig_res.update_yaxes(range=[-6, 6], row=1, col=2)
    fig_res.update_yaxes(range=[-6, 6], row=1, col=3)
    fig_res.update_yaxes(title_text='Strain (μstrain)', row=2, col=1, range=[-700, 700])
    fig_res.update_yaxes(range=[-700, 700], row=2, col=2)
    fig_res.update_yaxes(range=[-700, 700], row=2, col=3)
    fig_res.update_layout(plot_layout(
        height=440, showlegend=False,
    ))
    for ann in fig_res['layout']['annotations']:
        ann['font'] = dict(size=11, color=COLORS['text'], family=FONT)

    # =========== Figure 3: split into 3 separate figs so each gets its own description ===========
    t_long = d['t_long']
    t_win = (t_long >= 0) & (t_long < 120)
    periods = d['periods_fft']
    mask = (periods > 3) & (periods < 25)
    F_max = d['F_fft_smooth'][mask].max()
    S_max = d['S_smooth'][mask].max()

    # Panel 1: white-noise forcing time series
    fig_fft_force = go.Figure()
    fig_fft_force.add_trace(go.Scatter(
        x=t_long[t_win], y=d['F_white'][t_win]/1000,
        line=dict(color=COLORS['plot_2'], width=0.8),
        showlegend=False,
    ))
    fig_fft_force.update_layout(plot_layout(
        title=dict(text='Wave forcing F(t)  —  white-noise input  (kN)',
                    font=dict(size=12)),
        xaxis_title='Time (s)', yaxis_title='Force (kN)',
        height=220,
        margin=dict(l=60, r=20, t=40, b=40),
    ))

    # Panel 2: strain response time series
    fig_fft_strain = go.Figure()
    fig_fft_strain.add_trace(go.Scatter(
        x=t_long[t_win], y=d['strain_white'][t_win],
        line=dict(color=COLORS['alert'], width=0.8),
        showlegend=False,
    ))
    fig_fft_strain.update_layout(plot_layout(
        title=dict(text='Strain response ε(t)  —  the float\'s output  (μstrain)',
                    font=dict(size=12)),
        xaxis_title='Time (s)', yaxis_title='Strain (μstrain)',
        height=220,
        margin=dict(l=60, r=20, t=40, b=40),
    ))

    # Panel 3: FFT spectra
    fig_fft_spec = go.Figure()
    fig_fft_spec.add_trace(go.Scatter(
        x=periods[mask], y=d['F_fft_smooth'][mask]/F_max,
        line=dict(color=COLORS['plot_2'], width=1.5),
        name='Wave forcing FFT  (flat — white noise)',
    ))
    fig_fft_spec.add_trace(go.Scatter(
        x=periods[mask], y=d['S_smooth'][mask]/S_max,
        line=dict(color=COLORS['alert'], width=2),
        name=f'Strain response FFT  (peak @ {d["peak_period"]:.1f} s)',
    ))
    fig_fft_spec.add_vline(x=d['T_n'], line_dash='dash',
                            line_color=COLORS['text_dim'],
                            annotation_text=f"Tₙ={d['T_n']:.2f}s")
    fig_fft_spec.update_layout(plot_layout(
        title=dict(text='FFT spectra  —  forcing (blue) vs response (red)',
                    font=dict(size=12)),
        xaxis_title='Period (s)', yaxis_title='FFT magnitude (normalised)',
        xaxis=dict(range=[3, 25]),
        height=280,
        margin=dict(l=60, r=20, t=40, b=40),
        legend=dict(orientation='h', y=-0.30, x=0.5,
                     xanchor='center', font=dict(size=10)),
    ))

    # =========== Right column ===========
    right = html.Div([
        # === Section 1: SDOF abstraction ===
        section_header('1. The model — single-degree-of-freedom abstraction'),
        dcc.Markdown(
            "The float is physically a complex 6-DOF system (see the Overview "
            "tab), but for fatigue at the neck weld we only care about heave — "
            "vertical up-down motion driven by waves. So we collapse the entire "
            "system to one coordinate $x(t)$ and one equation of motion.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        physics_explanation(
            'Equation of motion (Newton\'s 2nd law for the lumped system)',
            'Three forces balance the wave forcing $F(t)$: inertia, damping, '
            'and restoring stiffness.',
        ),
        latex_equation(r'\underbrace{m\,\ddot{x}(t)}_{\text{inertia}} \;+\; \underbrace{c\,\dot{x}(t)}_{\text{damping}} \;+\; \underbrace{k\,x(t)}_{\text{restoring}} \;=\; F(t)'),
        dcc.Markdown(
            "The abstract spring-mass-damper sits on the left below. "
            "**The same physics on the actual float shape** sits on the "
            "right — same equation, same letters, mapped onto the parts "
            "of the WEC where each parameter physically lives:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '8px 0 10px 0'},
        ),
        html.Div([
            html.Div(
                dcc.Graph(figure=make_sdof_diagram(),
                           config={'displayModeBar': False}),
                style={'flex': '1 1 0', 'minWidth': '0'},
            ),
            html.Div(
                dcc.Graph(figure=make_float_sdof_figure(),
                           config={'displayModeBar': False}),
                style={'flex': '1 1 0', 'minWidth': '0'},
            ),
        ], style={'display': 'flex', 'gap': '12px',
                   'alignItems': 'flex-start', 'flexWrap': 'nowrap'}),

        # Legend for the SDOF symbols, rendered as compact HTML below
        # the two figures so nothing gets clipped inside the canvas.
        html.Div([
            html.Div('SDOF parameters', style={
                'fontSize': '12px', 'fontWeight': '600',
                'color': COLORS['text_dim'],
                'textTransform': 'uppercase',
                'letterSpacing': '0.05em',
                'margin': '0 0 8px 0',
            }),
            dcc.Markdown(
                "- $m$  — effective mass (structural + added water)\n"
                "- $k$  — restoring stiffness  ($\\rho\\,g\\,A_{wp}$ "
                "from buoyancy)\n"
                "- $c$  — damping  (radiated waves + viscous drag + PTO)\n"
                "- $F(t)$  — wave-induced vertical force\n"
                "- $x(t)$  — heave displacement  (the output)",
                mathjax=True,
                style={'fontSize': '13px', 'lineHeight': '1.7',
                        'margin': '0'},
            ),
        ], style={
            'background': '#F5F5F0',
            'border': f"1px solid {COLORS['border']}",
            'borderRadius': '4px',
            'padding': '10px 14px',
            'margin': '8px 0 16px 0',
        }),

        # === Section 2: Parameters with derivations ===
        section_header('2. The three parameters — values & derivations'),

        physics_explanation(
            'Effective mass $m$',
            'Includes structural mass + added-water mass (the co-moving water '
            'blob the float drags through the ocean) + coupled inertia from '
            'the submerged mass below.',
        ),
        dcc.Markdown(
            f"For this dashboard we take "
            f"$m \\approx {fmt_value(p_m)}$ kg "
            f"(≈ {p_m/1000:.0f} tons).",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        physics_explanation(
            'Structural stiffness $k$ — from hydrostatic buoyancy, NOT from the steel',
            'Push the float down by $\\Delta z$. It displaces an extra volume of '
            'water $A_{wp} \\cdot \\Delta z$ (where $A_{wp}$ is the waterplane '
            'area). Archimedes gives the restoring force. Geometry rules, not '
            'material — a wide flat float has high $k$, a narrow tall float has '
            'low $k$.',
        ),
        latex_equation(r'F_{\text{restore}} \;=\; \rho \, g \, A_{wp} \, \Delta z \qquad \Longrightarrow \qquad k \;=\; \rho \, g \, A_{wp}'),
        dcc.Markdown(
            f"Plugging in $\\rho = 1025$ kg/m³, $g = 9.8$ m/s², "
            f"$A_{{wp}} \\approx {A_wp_est:.1f}$ m² gives "
            f"$k \\approx {fmt_value(p_k)}$ N/m.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        physics_explanation(
            'Damping coefficient $c$ — lumps three sources',
            '(1) viscous water drag, (2) wave-radiation damping (the moving float '
            'radiates small waves of its own), (3) PTO damping (energy extraction '
            f'*is* damping). Chosen by reverse-engineering from a target damping '
            f'ratio $\\zeta = {p_zeta:.2f}$, typical for floating structures.',
        ),
        latex_equation(r'c \;=\; 2 \, \zeta \, \sqrt{m \, k}'),
        dcc.Markdown(
            f"With $m = {fmt_value(p_m)}$ kg, $k = {fmt_value(p_k)}$ N/m, "
            f"and $\\zeta = {p_zeta:.2f}$, that gives "
            f"$c = {fmt_value(p_c)}$ N·s/m.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # === Section 3: Derived quantities ===
        section_header('3. Derived quantities — natural frequency, damping ratio, Q'),

        physics_explanation(
            'Natural angular frequency $\\omega_n$ and natural period $T_n$',
            'The frequency the float oscillates at when disturbed and left alone. '
            f'$T_n \\approx {T_n:.1f}$ s sits squarely in the middle of typical '
            'ocean wave periods (8-16 s), so resonance is a real operational risk.',
        ),
        latex_equation(r'\omega_n \;=\; \sqrt{\dfrac{k}{m}} \qquad T_n \;=\; \dfrac{2\pi}{\omega_n}'),
        dcc.Markdown(
            f"With $k = {fmt_value(p_k)}$ N/m and $m = {fmt_value(p_m)}$ kg, "
            f"$\\omega_n = {omega_n:.3f}$ rad/s and "
            f"$T_n = {T_n:.2f}$ s.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        physics_explanation(
            'Damping ratio $\\zeta$ and quality factor $Q$',
            '$\\zeta$ is dimensionless. Light damping ($\\zeta \\ll 1$) gives a '
            'sharp resonance peak; heavy damping ($\\zeta \\to 1$) flattens it. '
            '$Q$ is the **resonance amplification** — at $\\omega = \\omega_n$, '
            'response amplitude is $Q$× larger than the static response to the '
            'same force.',
        ),
        latex_equation(r'\zeta \;=\; \dfrac{c}{2\sqrt{m\,k}} \qquad Q \;=\; \dfrac{1}{2\zeta}'),
        dcc.Markdown(
            f"With the values above, $\\zeta = {p_zeta:.2f}$ and "
            f"$Q = {Q:.1f}$  (when $\\zeta = {p_zeta:.2f}$). "
            f"Fatigue damage scales as stress³, so at resonance the "
            f"damage rate is roughly $Q^{{3}} \\approx {Q**3:.0f}\\times$ "
            f"per cycle compared to the off-resonance case.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # === Section 4: Transfer function ===
        section_header('4. The transfer function — frequency-response solution'),
        physics_explanation(
            'Derivation',
            'Assume sinusoidal forcing $F(t) = F_{0}\\, e^{i\\omega t}$ and '
            'sinusoidal response $x(t) = X\\, e^{i\\omega t}$. Substitute into '
            'the ODE; the exponential cancels; solve for $X/F_{0}$:',
        ),
        latex_equation(r'(-m\omega^{2} + ic\omega + k)\, X \, e^{i\omega t} \;=\; F_{0}\, e^{i\omega t}'),
        latex_equation(r'H(\omega) \;=\; \frac{X}{F_{0}} \;=\; \frac{1}{k - m\omega^{2} + ic\omega}'),
        latex_equation(r'|H(\omega)| \;=\; \frac{1}{\sqrt{(k - m\omega^{2})^{2} + (c\omega)^{2}}} \qquad \text{units: m/N}'),

        html.Div([
            dcc.Markdown(
                'At $\\omega = 0$ (static), the inertia and damping terms '
                'vanish — only the stiffness term remains. At '
                '$\\omega = \\omega_{n}$, the $(k - m\\omega^{2})$ term '
                'vanishes by definition of $\\omega_{n}$. So the two limit '
                'cases are simple:',
                mathjax=True,
                style={'fontSize': '13.5px', 'lineHeight': '1.65'},
            ),
        ], style={'margin': '12px 0 4px 0'}),
        latex_equation(r'|H(0)| \;=\; \dfrac{1}{k} \qquad |H(\omega_n)| \;=\; \dfrac{1}{c\,\omega_n} \qquad \dfrac{|H(\omega_n)|}{|H(0)|} \;=\; \dfrac{k}{c\,\omega_n} \;=\; \dfrac{1}{2\zeta} \;=\; Q'),
        dcc.Markdown(
            f"With the values from Section 2 ($k = {fmt_value(p_k)}$ N/m, "
            f"$c = {fmt_value(p_c)}$ N·s/m, $\\omega_n = {omega_n:.3f}$ "
            f"rad/s), the two limit cases evaluate to:\n\n"
            f"- Static: $|H(0)| = 1/k = {H_static_um_per_N:.0f}\\ "
            f"\\mu\\text{{m/N}}$.\n"
            f"- Resonance: $|H(\\omega_n)| = 1/(c\\,\\omega_n) = "
            f"{H_res_um_per_N:.0f}\\ \\mu\\text{{m/N}}$.\n"
            f"- Ratio = $Q = 1/(2\\zeta) = {Q:.1f}$  (when "
            f"$\\zeta = {p_zeta:.2f}$) — the float oscillates "
            f"**{Q:.1f} times harder** at resonance for the same "
            f"forcing amplitude, which matches the quality factor we "
            f"derived in Section 3.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Graph(figure=fig_tf, config={'displayModeBar': False}),

        expandable_note('Reading the table — what |H| means physically', [
            dcc.Markdown(
                f"The **|H(ω)|** column tells you how many micrometers "
                f"the float heaves per Newton of wave force, at that "
                f"frequency. A few examples:\n\n"
                f"- At resonance (ω = ωₙ), |H| = {H_res_um_per_N:.0f} μm/N. "
                f"A {F_demo_kN:.0f} kN wave force produces "
                f"{F_demo_N:.0f} × {H_res_um_per_N:.0f} μm = "
                f"**{F_demo_N*H_res_um_per_N/1e6:.2f} m of heave**.\n"
                f"- At the static limit (ω = 0), |H| = "
                f"{H_static_um_per_N:.0f} μm/N — the same "
                f"{F_demo_kN:.0f} kN force produces only "
                f"**{F_demo_N*H_static_um_per_N/1e6:.2f} m of heave**. "
                f"The resonance value is "
                f"**1/(2ζ) = {Q:.1f}× larger**  (when "
                f"ζ = {p_zeta:.2f}).",
                mathjax=True,
                style={'fontSize': '13px', 'lineHeight': '1.65'},
            ),
        ]),

        expandable_note('How heave converts to strain at the weld', [
            dcc.Markdown(
                f"To convert heave displacement to surface strain at the "
                f"neck weld we use a geometry factor "
                f"$\\kappa \\approx {kappa_us_per_m:.0f}\\ \\mu\\varepsilon$ "
                f"per meter of heave. This is an **assumed value** for "
                f"the stylised neck cross-section used throughout this "
                f"dashboard.",
                mathjax=True,
                style={'fontSize': '13px', 'lineHeight': '1.65'},
            ),
        ]),

        styled_table(
            headers=['Scenario', 'T_wave', 'ω (rad/s)', 'vs ωₙ',
                      '|H| (μm/N)', 'vs static',
                      f'Strain at F = {F_demo_kN:.0f} kN'],
            rows=[
                ['Static (DC limit)', '∞',     '0.000', '—',
                  f"{H_static_um_per_N:.1f}", '1.0× (baseline)',
                  f"{F_demo_N*H_static_um_per_N/1000:.0f} μstrain"],
                [scs[0]['label'].split(':')[0],  f"{scs[0]['T_wave']:.1f} s",
                  f"{2*np.pi/scs[0]['T_wave']:.3f}",  '3.14× ωₙ (fast)',
                  f"{scs[0]['H_val']:.1f}",
                  f"{scs[0]['H_val']/H_static_um_per_N:.2f}×",
                  f"{scs[0]['H_val']*F_demo_N/1000:.0f} μstrain"],
                [scs[1]['label'].split(':')[0],  f"{scs[1]['T_wave']:.1f} s",
                  f"{2*np.pi/scs[1]['T_wave']:.3f}",  '= ωₙ (RESONANCE)',
                  f"{scs[1]['H_val']:.1f}",
                  f"{scs[1]['H_val']/H_static_um_per_N:.1f}×",
                  f"{scs[1]['H_val']*F_demo_N/1000:.0f} μstrain"],
                [scs[2]['label'].split(':')[0],  f"{scs[2]['T_wave']:.1f} s",
                  f"{2*np.pi/scs[2]['T_wave']:.3f}",  '0.42× ωₙ (slow)',
                  f"{scs[2]['H_val']:.1f}",
                  f"{scs[2]['H_val']/H_static_um_per_N:.2f}×",
                  f"{scs[2]['H_val']*F_demo_N/1000:.0f} μstrain"],
            ],
            highlight_row=2,
        ),

        # === Section 5: FFT system identification (now BEFORE the demo) ===
        section_header('5. Get ω_n empirically from sensor data — '
                        'FFT of the strain response'),

        dcc.Markdown(
            "Section 3 gave us $\\omega_n$ from $\\sqrt{k/m}$. That's "
            "the **model-based** estimate; it depends on knowing $m$ "
            "and $k$ precisely. In real fleet operations we usually "
            "don't — biofouling adds mass, joint wear changes "
            "stiffness, and per-unit values drift over time. So we "
            "want a **second, data-driven estimate** that requires no "
            "model.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Markdown(
            "**The idea:** feed the float random forces and look at "
            "what frequency it prefers to oscillate at — that's its "
            "natural frequency, read directly off the strain spectrum.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        expandable_note('Why this works (the underlying theorem)', [
            dcc.Markdown(
                "For any linear system driven by random input with "
                "power spectrum $S_F(\\omega)$, the output power "
                "spectrum is the input spectrum times the squared "
                "transfer function:",
                mathjax=True,
                style={'fontSize': '13px', 'lineHeight': '1.65'},
            ),
            latex_equation(r'S_x(\omega) \;=\; |H(\omega)|^{2} \cdot S_F(\omega)'),
            dcc.Markdown(
                "When the input is **white noise** (flat $S_F$), the "
                "output spectrum traces $|H(\\omega)|^{2}$ directly — "
                "and $|H|^{2}$ has its peak at $\\omega = \\omega_n$. "
                "So picking the peak of the response spectrum *is* "
                "picking $\\omega_n$.",
                mathjax=True,
                style={'fontSize': '13px', 'lineHeight': '1.65'},
            ),
        ]),

        # === Panel 1: white-noise forcing ===
        dcc.Graph(figure=fig_fft_force, config={'displayModeBar': False}),
        dcc.Markdown(
            "**Forcing** is white noise — every wave frequency is "
            "present at the same amplitude. So in the frequency domain "
            "the input spectrum will be **flat**.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # === Panel 2: strain response ===
        dcc.Graph(figure=fig_fft_strain, config={'displayModeBar': False}),
        dcc.Markdown(
            "**Strain response** at the neck weld. The time series "
            "looks noisy, but already shows a dominant oscillation "
            "near $T_n$ — the float ignores most of the broadband "
            "input and resonates at its preferred frequency.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # === Panel 3: FFT spectra ===
        dcc.Graph(figure=fig_fft_spec, config={'displayModeBar': False}),
        dcc.Markdown(
            f"**FFT spectra.** The forcing spectrum (blue) is flat, "
            f"as expected for white noise. The response spectrum "
            f"(red) has a clear peak at $T = {d['peak_period']:.2f}$ s "
            f"— the float amplifies signals near its natural "
            f"frequency by a factor of "
            f"$Q = 1/(2\\zeta) = {Q:.1f}$ (when "
            f"$\\zeta = {p_zeta:.2f}$) and damps everything else. "
            f"**Reading the peak location off the red curve gives the "
            f"empirical estimate of $T_n$.**",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        dcc.Markdown(
            f"**Comparing the two estimates:**\n\n"
            f"- **Theoretical** (Section 3): "
            f"$T_n = 2\\pi\\sqrt{{m/k}} = {d['T_n']:.2f}$ s.\n"
            f"- **FFT of the strain response** (this section): "
            f"**{d['peak_period']:.2f} s**.\n\n"
            f"They agree, which is the sanity check we wanted. In real "
            f"fleet operation we only have the FFT estimate — and we "
            f"monitor it for drift.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # === Why FFT for fleet monitoring (collapsible) ===
        expandable_note('+ Why this matters for fleet monitoring', [
            dcc.Markdown(
                "Each deployed unit has slightly different mass, "
                "stiffness, and damping, and these drift over time — "
                "biofouling adds mass, water ingress adds mass, joint "
                "wear reduces stiffness. The natural frequency therefore "
                "cannot be predicted analytically and trusted over a "
                "20-year service life. Continuous strain telemetry plus "
                "a rolling FFT recovers each unit's natural frequency "
                "directly from data; a drift greater than ~3% from the "
                "unit's baseline is the alarm condition for "
                "structural-health monitoring.",
                style={'fontSize': '13px', 'lineHeight': '1.65'},
            ),
        ]),

        # === Section 6: Resonance demo (NOW after FFT identification) ===
        section_header('6. Resonance demo — same forcing, three different wave periods'),
        html.P(
            f"Now that we know ωₙ (both theoretically and from FFT), watch "
            f"what happens when wave period sweeps past Tₙ. Three "
            f"scenarios with identical wave-force amplitude ({F_demo_kN:.0f} "
            f"kN peak), different wave periods: 4 s (off-resonance, fast), "
            f"{T_n:.1f} s (at resonance), 30 s (slow, quasi-static). Top row "
            f"is the wave forcing (same amplitude in all three!); bottom "
            f"row is the strain response (dramatically different "
            f"amplitudes).",
            style={'fontSize': '13px', 'lineHeight': '1.6'},
        ),
        dcc.Graph(figure=fig_res, config={'displayModeBar': False}),

        expandable_note('How the table columns are computed', [
            dcc.Markdown(
                f"Each row uses one consistent set of physics:\n\n"
                f"- $\\sigma_{{\\text{{peak}}}}$: from $|H(\\omega)| \\cdot "
                f"F_{{\\text{{peak}}}} \\cdot \\kappa \\cdot 0.2$ "
                f"(MPa per μstrain), with $F_{{\\text{{peak}}}} = "
                f"{F_demo_kN:.0f}$ kN, "
                f"$\\kappa = {kappa_us_per_m:.0f}$ μstrain/m.\n"
                f"- **D / year**: from BS 7608 S-N curve + Miner's rule "
                f"(Strain & Fatigue tab steps 6-7), counting "
                f"$2\\cdot 3600\\cdot 24\\cdot 365 / T_{{\\text{{wave}}}}$ "
                f"cycles per year.\n"
                f"- **Years to D=1**: $1 / D_{{\\text{{year}}}}$. Compare to "
                f"the 20-year design life.\n"
                f"- **vs Yield / UTS**: $\\sigma_{{\\text{{peak}}}}$ against "
                f"$\\sigma_y/SF = "
                f"{DATA['fatigue']['material']['yield_str']/DATA['fatigue']['material']['yield_sf']:.0f}$ "
                f"MPa and "
                f"$\\sigma_{{\\text{{UTS}}}}/SF = "
                f"{DATA['fatigue']['material']['UTS']/DATA['fatigue']['material']['UTS_sf']:.0f}$ "
                f"MPa from the Strain & Fatigue tab Step 9.",
                mathjax=True,
                style={'fontSize': '13px', 'lineHeight': '1.65'},
            ),
        ]),

        _resonance_scenarios_table(d),

        # Reminder of the hard-limit allowables (in case the user has scrolled
        # past Strain & Fatigue Step 9). All numbers come from the
        # material dict in DATA['fatigue'], so they stay consistent.
        dcc.Markdown(
            f"**Reminder of the hard-limit allowables** (from Strain & "
            f"Fatigue tab, Step 9):\n\n"
            f"- Yield allowable: "
            f"$\\sigma_y / SF = "
            f"{DATA['fatigue']['material']['yield_str']:.0f}\\,/\\,"
            f"{DATA['fatigue']['material']['yield_sf']} = "
            f"{DATA['fatigue']['material']['yield_str']/DATA['fatigue']['material']['yield_sf']:.0f}$ "
            f"MPa\n"
            f"- UTS allowable: "
            f"$\\sigma_\\text{{UTS}} / SF = "
            f"{DATA['fatigue']['material']['UTS']:.0f}\\,/\\,"
            f"{DATA['fatigue']['material']['UTS_sf']} = "
            f"{DATA['fatigue']['material']['UTS']/DATA['fatigue']['material']['UTS_sf']:.0f}$ "
            f"MPa\n"
            f"- Fatigue: $D < 1.0$ over the 20-year design life "
            f"(BS 7608 S-N curve + Miner)",
            mathjax=True,
            style={'fontSize': '13px', 'lineHeight': '1.7',
                    'background': '#F5F5F0',
                    'border': f"1px solid {COLORS['border']}",
                    'borderRadius': '4px',
                    'padding': '10px 14px',
                    'margin': '12px 0 4px 0'},
        ),
    ])

    return html.Div([side, right], style=STYLES['two_col'])


def _resonance_scenarios_table(d):
    """Compute the resonance-demo scenario table from physics.

    Three wave periods (off-resonance, at resonance, quasi-static), same
    forcing amplitude p['F_demo_N']. For each, we evaluate:
      * peak heave from |H(ω)|·F_peak
      * peak strain from κ·heave  (κ = strain_per_meter from data)
      * peak stress from 0.2 MPa/μstrain
      * fatigue D/year via the BS 7608 S-N curve (Strain & Fatigue tab)
      * yield + UTS hard-limit checks vs allowable (Strain & Fatigue Step 9)

    Numbers update automatically if dashboard parameters change.
    """
    p = d['params']
    m, k, c = p['m'], p['k'], p['c']
    F_peak = p['F_demo_N']                           # forcing amplitude
    F_peak_kN = F_peak / 1000.0
    kappa = p['strain_per_meter']                    # μstrain per meter
    omega_n = d['omega_n']

    # Material / fatigue constants (mirror Strain & Fatigue tab)
    fat = DATA['fatigue']
    mat = fat['material']
    sigma_lim   = mat['sigma_limit']    # MPa (BS 7608 Class D × seawater)
    N_lim       = mat['N_limit']        # 1e7 anchor cycles
    m_sn        = mat['m_sn']           # 3
    yield_str   = mat['yield_str']
    yield_sf    = mat['yield_sf']
    UTS         = mat['UTS']
    UTS_sf      = mat['UTS_sf']
    sigma_y_allow   = yield_str / yield_sf
    sigma_uts_allow = UTS / UTS_sf

    sec_per_year = 365 * 24 * 3600.0

    scenario_specs = [
        ('A: Off-resonance  (T = 4 s)',          4.0),
        ('B: AT RESONANCE   (T = 12.57 s)',      2*np.pi/omega_n),
        ('C: Quasi-static   (T = 30 s)',         30.0),
    ]

    rows = []
    for label, T_wave in scenario_specs:
        omega = 2*np.pi / T_wave
        # |H(ω)| in m/N
        H_mag = 1.0 / np.sqrt((k - m*omega**2)**2 + (c*omega)**2)
        x_peak_m       = F_peak * H_mag                     # peak heave (m)
        eps_peak_us    = x_peak_m * kappa                   # peak strain (μstrain)
        sigma_peak_MPa = eps_peak_us * 0.2                  # MPa  (rule of thumb)

        # Fatigue D/year: every wave cycle is one stress cycle of full range 2·σ_peak.
        # Conservative: count the peak-to-peak amplitude as the "range" σ in the S-N
        # curve, which is exactly what BS 7608 assumes for sinusoidal loading.
        cycles_per_year = sec_per_year / T_wave
        if sigma_peak_MPa > sigma_lim:
            N_fail = N_lim * (sigma_lim / sigma_peak_MPa) ** m_sn
            d_per_cycle = 1.0 / N_fail
        else:
            d_per_cycle = 0.0
        D_year = cycles_per_year * d_per_cycle
        years_to_fail = (1.0 / D_year) if D_year > 0 else float('inf')

        # Hard-limit checks
        yield_pass = sigma_peak_MPa < sigma_y_allow
        uts_pass   = sigma_peak_MPa < sigma_uts_allow
        fatigue_pass = D_year * 20 < 1.0
        overall_pass = fatigue_pass and yield_pass and uts_pass

        years_str = (f'{years_to_fail:.1f} yr'
                      if np.isfinite(years_to_fail) else 'infinite')

        def _pass_span(passed, allow_val):
            txt = ('✓ ' + f'{allow_val:.0f} MPa') if passed else (
                'FAIL  ' + f'(>{allow_val:.0f} MPa)')
            col = COLORS['accent_2'] if passed else COLORS['alert']
            return html.Span(txt, style={'fontWeight': '700',
                                          'color': col})

        rows.append([
            label,
            f'{sigma_peak_MPa:.1f} MPa',
            (fmt_value(D_year) if D_year > 0 else '0'),
            years_str,
            _pass_span(yield_pass, sigma_y_allow),
            _pass_span(uts_pass,   sigma_uts_allow),
            html.Span('SAFE' if overall_pass else 'FAILS',
                       style={'fontWeight': '700',
                               'color': (COLORS['accent_2'] if overall_pass
                                         else COLORS['alert'])}),
        ])

    return styled_table(
        headers=[f'Scenario  (forcing F = {F_peak_kN:.0f} kN)',
                  'Peak σ', 'D / year', 'Years to D = 1',
                  'vs Yield/SF', 'vs UTS/SF',
                  'Verdict'],
        rows=rows,
        highlight_row=1,
    )


def tab_pressure():
    """Tab 4: Pressure sensors — wrapper with PTO baseline / swing
    sliders + dynamic container for parameter-dependent content."""
    d_init = DATA['pressure']

    slider_box_style = {
        'background': '#F5F5F0',
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '4px',
        'padding': '14px 18px',
        'margin': '0 0 20px 0',
    }
    slider_label_style = {
        'fontSize': '12px', 'fontWeight': '600',
        'color': COLORS['text_dim'],
        'textTransform': 'uppercase',
        'letterSpacing': '0.04em',
        'margin': '0 0 6px 0',
    }

    def _slider_block(label_html, slider):
        return html.Div([
            html.Div(label_html, style=slider_label_style),
            slider,
        ], style={'flex': '1 1 0', 'minWidth': '180px'})

    slider_base = dcc.Slider(
        id='pre-pto-base', min=10, max=100, step=10,
        value=int(d_init['pto_baseline_bar']),
        marks={v: f'{v}' for v in [10, 30, 50, 70, 100]},
        tooltip={'placement': 'bottom', 'always_visible': False},
    )
    slider_swing = dcc.Slider(
        id='pre-pto-swing', min=1, max=10, step=1,
        value=int(d_init['pto_swing_bar']),
        marks={v: f'{v}' for v in [1, 3, 5, 7, 10]},
        tooltip={'placement': 'bottom', 'always_visible': False},
    )

    sliders_panel = html.Div([
        html.Div('Tune the PTO design — pressure-driven numbers update '
                  'live', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text'],
            'margin': '0 0 12px 0',
        }),
        html.Div([
            _slider_block('PTO baseline pressure (bar)', slider_base),
            _slider_block('PTO swing amplitude  ΔP (bar)', slider_swing),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}),
    ], style=slider_box_style)

    return html.Div([
        tab_header(
            'Pressure — five channels, three engineering questions',
            "- **What does the ocean look like?** External hull sensors "
            "at three depths recover the wave amplitude via linear wave "
            "theory and cross-check against the wave-power flux.\n"
            "- **How much power are we generating?** Internal "
            "hydraulic-chamber pressure → energy per stroke → "
            "wave-to-wire electrical output and conversion efficiency.\n"
            "- **Is the float intact?** Dry-cabin and hull-bottom depth "
            "sensors give a leak / biofouling / turbine-fault "
            "fingerprint table read across all five channels at once."
        ),
        sliders_panel,
        dcc.Loading(
            id='pressure-loading',
            type='circle',
            color=COLORS['accent'],
            children=html.Div(id='pressure-output',
                                children=_tab_pressure_body(d_init)),
        ),
    ])


def _tab_pressure_body(d):
    """Parameter-dependent content of the Pressure tab."""
    # ==================== Wave / linear-theory parameters ====================
    T_wave = d['T_wave']
    omega = 2 * np.pi / T_wave
    g = 9.81
    h = 50.0                    # total water depth, m
    k_wave = omega ** 2 / g     # deep-water dispersion: ω² = gk
    wavelength = 2 * np.pi / k_wave
    kh = k_wave * h

    # Attenuation table (depths used in the 'real' notebook output)
    depths_for_table = [0, 1, 3, 6, 10, 25, 50]
    atten_table = [(z, np.cosh(k_wave * (h - z)) / np.cosh(k_wave * h))
                    for z in depths_for_table]

    # ==================== Power-calculation parameters ====================
    piston_area_m2     = 0.5
    stroke_length_m    = 0.5
    V_stroke_m3        = piston_area_m2 * stroke_length_m       # 0.25 m³
    eta_turbine        = 0.70
    pressure_swing_bar = d['pto_swing_bar']
    pressure_swing_Pa  = pressure_swing_bar * 1e5
    strokes_per_sec    = 2.0 / T_wave                            # push + pull
    E_stroke_kJ        = pressure_swing_Pa * V_stroke_m3 / 1000
    P_hyd_kW           = E_stroke_kJ * strokes_per_sec
    P_elec_kW          = P_hyd_kW * eta_turbine
    E_annual_MWh       = P_elec_kW * 8760 / 1000

    # ==================== Wave-to-wire efficiency =========================
    # Wave power per unit crest length for deep-water linear waves:
    #     J = ρ g² A² T / (8π)
    rho_sea = 1025.0
    g_const = 9.81
    A_wave  = 1.0                                                # measured (Sensor 1)
    P_wave_per_m_kW   = rho_sea * g_const**2 * A_wave**2 * T_wave / (8 * np.pi) / 1000
    capture_width_m   = 2.5                                      # float diameter scale
    P_wave_total_kW   = P_wave_per_m_kW * capture_width_m
    wave_to_wire_pct  = 100.0 * P_elec_kW / P_wave_total_kW

    # ==================== SIDEBAR — concise tab overview ====================
    side = sidebar(
        'Pressure Sensors — 5 channels',
        """
This tab walks through the five pressure channels on the device and
shows what each one is for, along with the engineering numbers each
one produces.

**The five sensors:**

1. **External hull** (×3 at depths 1, 3, 6 m) — measures wave loading
2. **Internal PTO chamber** — the energy-generating signal
3. **Differential across turbine** — cross-check on PTO
4. **Dry-side cabin** — leak detection
5. **Depth / draft** — hull-bottom reference
""",
        """
The pressure tab demonstrates the end-to-end chain from raw pressure
signal to engineering decision: wave loading → power calculation →
wave-to-wire efficiency → fault diagnosis. Each section below
includes the setup parameters used and the numbers actually computed
from the data.
""",
        {},
    )

    # ==================== FIGURES ====================
    palette = [COLORS['plot_1'], COLORS['plot_2'], COLORS['plot_3']]

    # --- Panel 1: external hull pressure ---
    fig1 = go.Figure()
    for (lbl, P), color in zip(d['hull'].items(), palette):
        fig1.add_trace(go.Scatter(x=d['t'], y=P, name=f'Hull, {lbl}',
                                    line=dict(color=color, width=1.2)))
    fig1.update_layout(plot_layout(
        title=dict(text='External hull pressure  P(t, z)  —  wave-induced '
                         f'oscillation at depths {", ".join(d["hull"].keys())}',
                    font=dict(size=13)),
        xaxis_title='Time (s)', yaxis_title='Pressure (bar)',
        height=280, legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
    ))

    # --- Panel 2: internal PTO pressure ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=d['t'], y=d['P_internal'], showlegend=False,
                                line=dict(color=COLORS['plot_2'], width=1.2)))
    fig2.add_hline(y=d['pto_baseline_bar'], line_dash='dot',
                    line_color=COLORS['text_dim'],
                    annotation_text=f"{d['pto_baseline_bar']:.0f} bar baseline",
                    annotation_position='top right')
    fig2.update_layout(plot_layout(
        title=dict(text=f"Internal PTO chamber pressure  —  wave-driven "
                         f"±{d['pto_swing_bar']:.0f} bar swing around the "
                         f"{d['pto_baseline_bar']:.0f} bar baseline",
                    font=dict(size=13)),
        xaxis_title='Time (s)', yaxis_title='Pressure (bar)',
        height=280,
    ))

    # --- Panel 3: differential ---
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=d['t'], y=d['P_diff'], showlegend=False,
                                line=dict(color=COLORS['plot_3'], width=1.2)))
    fig3.update_layout(plot_layout(
        title=dict(text='Differential pressure across turbine  ΔP = P_in − P_out',
                    font=dict(size=13)),
        xaxis_title='Time (s)', yaxis_title='ΔP (bar)',
        height=240,
    ))

    # --- Panel 4: dry cabin ---
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=d['t'], y=d['P_cabin'], showlegend=False,
                                line=dict(color=COLORS['plot_2'], width=1.2)))
    fig4.add_vline(x=40, line_dash='dash', line_color=COLORS['alert'],
                   annotation_text='leak begins', annotation_position='top right')
    fig4.add_hline(y=1.013, line_dash='dot', line_color=COLORS['text_dim'],
                   annotation_text='1 atm baseline', annotation_position='bottom right')
    fig4.update_layout(plot_layout(
        title=dict(text='Dry-side cabin pressure  —  constant near 1 atm; '
                         'rise indicates seal failure',
                    font=dict(size=13)),
        xaxis_title='Time (s)', yaxis_title='Pressure (bar)',
        height=240,
    ))

    # --- Panel 5: depth sensor ---
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=d['t'], y=d['P_depth'], showlegend=False,
                                line=dict(color=COLORS['plot_1'], width=1.2)))
    fig5.update_layout(plot_layout(
        title=dict(text='Depth / draft sensor  —  static (ρg·d_draft) + '
                         'dynamic wave-induced oscillation',
                    font=dict(size=13)),
        xaxis_title='Time (s)', yaxis_title='Pressure (bar)',
        height=240,
    ))

    # ==================== MAIN — background + 5 panels ====================
    plots = html.Div([
        # === Background: linear wave theory ===
        section_header('Background — linear wave theory'),
        dcc.Markdown(
            "Wave-induced pressure at depth $z$ below the surface (with the "
            "seabed at depth $h$) follows from linear wave theory:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'P_{\text{dyn}}(z, t) \;=\; \rho \, g \, A \, '
            r'\frac{\cosh\!\big(k(h-z)\big)}{\cosh(kh)} '
            r'\cos(\omega t)'
        ),
        dcc.Markdown(
            "where $\\rho$ = seawater density (1025 kg/m³), $A$ = wave "
            "amplitude, $k$ = wavenumber, $\\omega$ = angular frequency. "
            "The dispersion relation in deep water gives "
            "$\\omega^{2} = gk$, so the wavelength is "
            "$\\lambda = 2\\pi g / \\omega^{2}$.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        section_header('Wave parameters for this scenario'),
        styled_table(
            headers=['Quantity', 'Symbol', 'Value'],
            rows=[
                ['Wave amplitude',        'A',     f'{A_wave:.1f} m'],
                ['Wave period',           'T',     f'{T_wave:.1f} s'],
                ['Angular frequency',     'ω',     f'{omega:.3f} rad/s'],
                ['Wavenumber',            'k',     f'{k_wave:.4f} 1/m'],
                ['Wavelength',            'λ',     f'{wavelength:.1f} m'],
                ['Water depth',           'h',     f'{h:.0f} m'],
                ['Dimensionless depth',   'kh',    f'{kh:.2f}  (intermediate-water)'],
                ['Sampling frequency',    'f_s',   '10 Hz'],
            ],
        ),

        # === Wave power per unit crest length (also linear-theory) ===
        section_header('Wave power — what the ocean delivers'),
        dcc.Markdown(
            "From the same linear wave theory, the wave power flux per unit "
            "length of crest (deep-water limit) is — note the symbol "
            "$J$, the oceanographic convention to avoid clashing with "
            "pressure $P$:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'J \;=\; \dfrac{\rho \, g^{2} \, A^{2} \, T}{8\pi} \qquad \text{(W per meter of wave crest)}'),
        dcc.Markdown(
            f"For this scenario (A = {A_wave:.1f} m, T = {T_wave:.1f} s), "
            f"that gives **{P_wave_per_m_kW:.1f} kW per meter of crest** — "
            f"matching the textbook figure for the California coast "
            f"quoted in the Overview tab.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        # ============= SENSOR 1: EXTERNAL HULL =============
        section_header('Sensor 1 — External hull pressure (3 sensors)'),
        explanation_block([
            "Three pressure sensors mounted on the outer hull at depths ",
            html.Strong("1 m, 3 m, and 6 m"),
            ". They measure wave loading on the structure — the engineering "
            "input for fatigue and digital-twin force models.",
        ]),
        dcc.Graph(figure=fig1, config={'displayModeBar': False}),

        html.Div('Setup — pressure attenuation by depth', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '20px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        styled_table(
            headers=['Depth z (m)', 'Attenuation cosh(k(h−z))/cosh(kh)', '% of surface'],
            rows=[[f'{z}', f'{a:.3f}', f'{a*100:.0f}%']
                   for z, a in atten_table],
        ),

        # ============= SENSOR 2: INTERNAL PTO — POWER CALCULATION =============
        section_header('Sensor 2 — Internal PTO chamber pressure (wave-to-wire energy conversion)'),
        explanation_block(
            dcc.Markdown(
                "This is where the wave becomes electricity. Inside the "
                "hull, a hydraulic cylinder converts the wave-driven "
                "heave motion into pressurised working fluid.",
                style={'fontSize': '13.5px', 'lineHeight': '1.65', 'margin': '0'},
            ),
        ),
        dcc.Graph(figure=fig2, config={'displayModeBar': False}),
        dcc.Markdown(
            f"**Reading the plot:**\n"
            f"- The **baseline at {d['pto_baseline_bar']:.0f} bar** is "
            f"the standing pressure the hydraulic system holds when no "
            f"power is being generated.\n"
            f"- The **±{d['pto_swing_bar']:.0f} bar swing around that "
            f"baseline** is what drives the turbine. The wave pushes "
            f"the piston up and down; each stroke compresses then "
            f"releases the working fluid.\n\n"
            f"The {d['pto_baseline_bar']:.0f} bar baseline and "
            f"{d['pto_swing_bar']:.0f} bar swing in this dashboard are "
            f"**assumed values** (stylised low-pressure example), not "
            f"measurements from any specific WEC. Real WEC hydraulic-PTO "
            f"architectures vary widely.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),

        html.Div('Setup — design parameters (notional)', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '20px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        styled_table(
            headers=['Parameter', 'Symbol', 'Value'],
            rows=[
                ['Piston area  (internal)',
                  'A_piston', f'{piston_area_m2:.2f} m²'],
                ['Stroke length  (internal)',
                  'L_stroke', f'{stroke_length_m:.2f} m'],
                ['Volume per stroke  (= A · L)',
                  'V_stroke', f'{V_stroke_m3:.2f} m³'],
                ['Turbine + generator efficiency',
                  'η',        f'{eta_turbine*100:.0f}%'],
                ['Capture width  (external)',
                  'w_c',      f'{capture_width_m:.1f} m  (float diameter scale)'],
            ],
        ),

        # === Geometry schematic — moved here, AFTER the parameter table ===
        html.Div('Geometry — what the parameters mean', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '20px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        dcc.Markdown(
            "**Capture width** is an *external* hydrodynamic property — "
            "how much of the wave crest the float interacts with. "
            "**Piston area** and **stroke length** are *internal* "
            "mechanical properties of the hydraulic cylinder. The "
            "schematic clarifies the distinction:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Graph(figure=make_pto_geometry_figure(),
                   config={'displayModeBar': False}),

        html.Div('Measured from Sensor 2 plot above', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '18px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        styled_table(
            headers=['Quantity', 'Symbol', 'Value'],
            rows=[
                ['Pressure swing',     'ΔP',  f'{pressure_swing_bar:.1f} bar  '
                                              f'({pressure_swing_Pa:,.0f} Pa)'],
                ['Strokes per second', 'f_s', f'{strokes_per_sec:.3f}  '
                                              f'(= 2 / T)'],
            ],
        ),

        html.Div('Calculated output power', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['accent_2'], 'margin': '18px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        styled_table(
            headers=['Quantity', 'Formula', 'Value'],
            rows=[
                ['Energy per stroke',  'ΔP · V_stroke',
                  f'{E_stroke_kJ:.1f} kJ'],
                ['Hydraulic power',    'E_stroke · f_s',
                  f'{P_hyd_kW:.1f} kW'],
                ['Electrical power',   'P_hyd · η',
                  html.Span(f'{P_elec_kW:.2f} kW',
                             style={'fontWeight': '700', 'color': COLORS['accent_2']})],
                ['Annual energy',      'P_elec · 8760 h',
                  html.Span(f'{E_annual_MWh:.1f} MWh / year',
                             style={'fontWeight': '700', 'color': COLORS['accent_2']})],
            ],
        ),

        # === Wave-to-wire efficiency ===
        html.Div('Wave-to-wire efficiency — input vs output', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['warning'], 'margin': '24px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        dcc.Markdown(
            f"The Background section already computed the wave power "
            f"flux per meter of crest: **J = {P_wave_per_m_kW:.1f} kW/m**. "
            f"To get the wave power the float actually **intercepts**, "
            f"multiply by the capture width "
            f"$w_c \\approx {capture_width_m:.1f}$ m from the schematic "
            f"above:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        styled_table(
            headers=['Quantity', 'Formula', 'Value'],
            rows=[
                ['Wave power flux',
                  'J  (from Background)',
                  f'{P_wave_per_m_kW:.1f} kW / m'],
                ['Capture width',
                  'w_c  (geometry)',
                  f'{capture_width_m:.1f} m'],
                ['Intercepted wave power',
                  'J · w_c',
                  f'{P_wave_total_kW:.1f} kW'],
                ['Electrical output  (from above)',
                  '—',
                  f'{P_elec_kW:.2f} kW'],
                ['Wave-to-wire efficiency',
                  'P_elec / (J · w_c)',
                  html.Span(f'{wave_to_wire_pct:.1f} %',
                             style={'fontWeight': '700',
                                     'color': COLORS['warning']})],
            ],
        ),

        # ============= SENSOR 3: DIFFERENTIAL =============
        section_header('Sensor 3 — Differential pressure across turbine'),
        explanation_block(
            dcc.Markdown(
                "Pressure difference between the **inlet** (high-pressure "
                "side, upstream of the turbine) and the **outlet** "
                "(low-pressure side, downstream): "
                "$\\Delta P = P_{\\text{in}} - P_{\\text{out}}$. Acts as an "
                "independent cross-check on Sensor 2 — under normal operation "
                "the two should track each other, so a divergence between "
                "them is a turbine or seal-degradation signal.",
                mathjax=True,
                style={'fontSize': '13.5px', 'lineHeight': '1.65', 'margin': '0'},
            ),
        ),
        dcc.Graph(figure=fig3, config={'displayModeBar': False}),

        # ============= SENSOR 4: DRY CABIN — LEAK DETECTION =============
        section_header('Sensor 4 — Dry-side cabin pressure (leak detection)'),
        explanation_block(
            "Inside the sealed dry chamber housing the electronics. Should "
            "sit constant at atmospheric pressure (~1.013 bar)."
        ),
        dcc.Graph(figure=fig4, config={'displayModeBar': False}),

        # ============= SENSOR 5: DEPTH SENSOR =============
        section_header('Sensor 5 — Depth / draft sensor at hull bottom'),
        explanation_block(
            dcc.Markdown(
                "Mounted at the bottom of the hull. The sensor measures "
                "the **total water pressure** pressing on it, which has "
                "two parts:",
                mathjax=True,
                style={'fontSize': '13.5px', 'lineHeight': '1.65', 'margin': '0'},
            ),
        ),
        dcc.Markdown(
            "- **Static** part — constant, from the column of water "
            "above the sensor; tells you the float's draft.\n"
            "- **Dynamic** part — wave-induced; same attenuation "
            "formula as Sensor 1, evaluated at the sensor's depth.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(r'P_{\text{total}}(t) \;=\; \underbrace{\rho \, g \, d_{\text{draft}}}_{\text{static}} \;+\; \underbrace{\rho \, g \, A \, \dfrac{\cosh\!\big(k(h-d_{\text{draft}})\big)}{\cosh(kh)} \, \cos(\omega t)}_{\text{dynamic}}'),
        dcc.Markdown(
            "Why the plot looks like it does:\n\n"
            "- The **static** part puts the trace on a non-zero baseline "
            "— that baseline directly equals $\\rho g d_{\\text{draft}}$, "
            "so you read the draft straight off it.\n"
            "- The **dynamic** part adds wave-like oscillation on top, "
            "attenuated according to the sensor's depth.\n\n"
            "**What can shift the baseline over time:**\n"
            "- **Biofouling growth** → added weight → float sits lower → "
            "static baseline increases.\n"
            "- **Ballast or trim shifts** → static baseline changes.\n"
            "- **Mooring drift** (in moored deployments) → can show up "
            "in the time series.",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Graph(figure=fig5, config={'displayModeBar': False}),

        # ============= MULTI-SENSOR DIAGNOSIS =============
        section_header('Multi-sensor diagnosis — fault fingerprints across channels'),
        dcc.Markdown(
            "Different faults affect different sensors in different ways. "
            "Reading the five channels together turns ambiguous symptoms "
            "into a specific diagnosis:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        styled_table(
            headers=['Fault scenario', 'S1: hull', 'S2: PTO', 'S3: ΔP',
                      'S4: cabin', 'S5: depth'],
            rows=[
                ['Normal operation',
                  'oscillates with wave',
                  f"{d['pto_baseline_bar']:.0f} ± "
                  f"{d['pto_swing_bar']:.0f} bar",
                  'nominal swing',
                  '1.013 bar (flat)',  'flat / wave-driven'],
                ['Biofouling on hull',
                  '—',                   '—',          '—',
                  '—',                   html.Span('mean rises (heavier)',
                                                     style={'color': COLORS['alert']})],
                ['PTO seal leak',
                  '—',
                  html.Span(f"mean drops below {d['pto_baseline_bar']:.0f}",
                             style={'color': COLORS['alert']}),
                  html.Span('mean drops',
                             style={'color': COLORS['alert']}),
                  '—',                   '—'],
                ['Turbine stuck / blocked',
                  '—',
                  html.Span('pressure builds up',
                             style={'color': COLORS['alert']}),
                  html.Span('drops toward 0',
                             style={'color': COLORS['alert']}),
                  '—',                   '—'],
                ['Dry-cabin seal leak',
                  '—', '—', '—',
                  html.Span('slow rise above 1 atm',
                             style={'color': COLORS['alert']}),
                  '—'],
            ],
        ),
    ], style=STYLES['plots'])

    return html.Div([side, plots], style=STYLES['two_col'])


def tab_adcp():
    """Tab 5: ADCP — wrapper with current-model sliders + dynamic body."""
    d_init = DATA['adcp']

    slider_box_style = {
        'background': '#F5F5F0',
        'border': f"1px solid {COLORS['border']}",
        'borderRadius': '4px',
        'padding': '14px 18px',
        'margin': '0 0 20px 0',
    }
    slider_label_style = {
        'fontSize': '12px', 'fontWeight': '600',
        'color': COLORS['text_dim'],
        'textTransform': 'uppercase',
        'letterSpacing': '0.04em',
        'margin': '0 0 6px 0',
    }
    def _slider_block(label_text, slider):
        return html.Div([
            html.Div(label_text, style=slider_label_style),
            slider,
        ], style={'flex': '1 1 0', 'minWidth': '170px'})

    # M2 tidal amplitude (m/s)
    slider_M2 = dcc.Slider(id='adcp-M2', min=0.05, max=0.50, step=0.05,
                            value=d_init['M2_amplitude'],
                            marks={v/100: f'{v/100:.2f}' for v in [5, 15, 25, 35, 50]},
                            tooltip={'placement': 'bottom', 'always_visible': False})
    # Wind surface speed u₀ (m/s)
    slider_u0 = dcc.Slider(id='adcp-u0', min=0.0, max=0.30, step=0.03,
                            value=d_init['wind_surface'],
                            marks={v/100: f'{v/100:.2f}' for v in [0, 6, 12, 18, 24, 30]},
                            tooltip={'placement': 'bottom', 'always_visible': False})
    # Wind e-folding depth L_e (m)
    slider_Le = dcc.Slider(id='adcp-Le', min=2, max=20, step=2,
                            value=int(d_init['wind_e_fold']),
                            marks={v: f'{v}' for v in [2, 6, 10, 14, 20]},
                            tooltip={'placement': 'bottom', 'always_visible': False})
    # BBL thickness (m)
    slider_bbl = dcc.Slider(id='adcp-bbl', min=1, max=15, step=2,
                              value=int(d_init['bbl_thickness']),
                              marks={v: f'{v}' for v in [1, 5, 9, 13, 15]},
                              tooltip={'placement': 'bottom', 'always_visible': False})
    # Surface coupling factor
    slider_sc = dcc.Slider(id='adcp-sc', min=0.0, max=1.0, step=0.1,
                            value=d_init['surface_coupling'],
                            marks={v/10: f'{v/10:.1f}' for v in [0, 3, 6, 10]},
                            tooltip={'placement': 'bottom', 'always_visible': False})

    sliders_panel = html.Div([
        html.Div('Tune the simulated current field and the float\'s '
                  'coupling to the surface — every heat-map, profile and '
                  'residual updates live', style={
            'fontSize': '13px', 'fontWeight': '600',
            'color': COLORS['text'],
            'margin': '0 0 12px 0',
        }),
        html.Div([
            _slider_block('M2 tide amplitude  A_M2  (m/s)',     slider_M2),
            _slider_block('Wind surface speed  u₀  (m/s)',       slider_u0),
            _slider_block('Wind e-folding depth  L_e  (m)',      slider_Le),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap'}),
        html.Div([
            _slider_block('Bottom-boundary-layer thickness  (m)', slider_bbl),
            _slider_block('Surface coupling factor',             slider_sc),
        ], style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap',
                   'marginTop': '12px'}),
    ], style=slider_box_style)

    return html.Div([
        tab_header(
            'ADCP — measuring the water current under a drifting float',
            "- **Physics-based simulation** of a realistic ocean current "
            "(tidal flow + wind-driven surface layer + bottom boundary "
            "layer) sampled by a hull-mounted, downward-looking acoustic "
            "current profiler.\n"
            "- **Platform-motion correction**: a four-step GPS-fusion "
            "pipeline that subtracts the float's own velocity from the "
            "raw readings to recover the absolute current.\n"
            "- **Output**: depth-binned absolute current profile, with "
            "the residual error setting the noise floor of the corrected "
            "measurement."
        ),
        sliders_panel,
        dcc.Loading(
            id='adcp-loading',
            type='circle',
            color=COLORS['accent'],
            children=html.Div(id='adcp-output',
                                children=_tab_adcp_body(d_init)),
        ),
    ])


def _tab_adcp_body(d):
    """Parameter-dependent content of the ADCP tab."""

    # ===== Figure: 3 heatmaps + 1 line plot (4 panels) =====
    # Panel 1: heatmap of true u(t, z)
    # Panel 2: heatmap of raw ADCP measurement (biased)
    # Panel 3: line plot of device velocity from GPS (1-D)
    # Panel 4: heatmap of corrected u(t, z)

    vmin, vmax = -0.5, 0.5
    cscale = 'RdBu_r'

    def heatmap_plot(z_data, title):
        f = go.Figure(go.Heatmap(
            x=d['t_hr'], y=d['depths'], z=z_data.T,
            colorscale=cscale, zmid=0, zmin=vmin, zmax=vmax,
            colorbar=dict(title='u (m/s)', thickness=12),
        ))
        f.update_layout(plot_layout(
            title=dict(text=title, font=dict(size=14)),
            xaxis_title='Time (hr)', yaxis_title='Depth (m)',
            yaxis=dict(autorange='reversed'),
            height=240,
            margin=dict(l=60, r=20, t=40, b=40),
        ))
        return f

    fig_p1 = heatmap_plot(d['u_true'],   'u(t, z)   —   m/s')
    fig_p2 = heatmap_plot(d['u_raw'],    'u_raw(t, z)   —   m/s')

    fig_p3 = go.Figure()
    fig_p3.add_trace(go.Scatter(
        x=d['t_hr'], y=d['u_device_gps'],
        line=dict(color=COLORS['plot_3'], width=1.5),
        showlegend=False,
    ))
    fig_p3.update_layout(plot_layout(
        title=dict(text='u_device(t)   —   m/s', font=dict(size=13)),
        xaxis_title='Time (hr)', yaxis_title='u_device (m/s)',
        height=200,
        margin=dict(l=60, r=20, t=40, b=40),
    ))

    fig_p4 = heatmap_plot(d['u_corrected'], 'u_corrected(t, z)   —   m/s')

    # ===== Sidebar — WHOI-style principle =====
    side = sidebar(
        'ADCP — Acoustic Doppler Current Profiler',
        """
**How it works.** The ADCP measures water current with sound, using the
Doppler effect — the same principle that makes a passing car sound
high-pitched as it approaches and lower as it recedes.

The instrument transmits "pings" of sound at a constant frequency.
The pings ricochet off particles suspended in the water and reflect
back. Particles moving toward the instrument send back higher-frequency
waves; particles moving away send back lower-frequency waves. The
difference between transmitted and received frequency — the Doppler
shift — is converted to water velocity.

Sound waves that hit particles further away take longer to return.
By measuring this travel time, the ADCP measures current speed at
many different depths from a single ping — the "current profile" in
the name.
""",
        "",
        {},
    )
    # Append source link
    side = html.Div([
        side,
        html.Div([
            html.Span('Source: ', style={
                'fontSize': '11px', 'color': COLORS['text_dim'],
            }),
            html.A(
                'WHOI',
                href='https://www.whoi.edu/what-we-do/explore/instruments/'
                     'instruments-sensors-samplers/'
                     'acoustic-doppler-current-profiler-adcp/',
                target='_blank',
                style={
                    'fontSize':       '11px',
                    'color':          COLORS['accent'],
                    'textDecoration': 'underline',
                },
            ),
        ], style={'padding': '8px 16px 16px 16px'}),
    ], style={**STYLES['sidebar'], 'padding': '0'})

    # ===== Plots column =====
    plots = html.Div([
        section_header('Setup — instrument geometry, current model, and fusion'),

        html.Div('Instrument geometry', style={
            'fontSize': '12px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '12px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        dcc.Graph(
            figure=make_adcp_geometry_figure(
                h_water=float(d['h_water']),
                adcp_depth=float(d['adcp_depth']),
                depth_bin_top=float(d['depths'][0]),
                depth_bin_bottom=float(d['depths'][-1]),
                n_bins=int(d['n_bins']),
            ),
            config={'displayModeBar': False},
        ),
        styled_table(
            headers=['Quantity', 'Value'],
            rows=[
                ['Local water depth  h',
                  f'{d["h_water"]:.1f} m'],
                ['ADCP transducer depth  (hull bottom)',
                  f'{d["adcp_depth"]:.1f} m below surface'],
                ['ADCP depth bins',
                  f'{int(d["depths"][0])} m to {int(d["depths"][-1])} m below '
                  f'surface  ({d["n_bins"]} bins, {d["bin_thickness"]:.0f} m each)'],
                ['Sample period',
                  f'{d["sample_period_min"]:.0f}-min averages  '
                  f'({d["n_samples"]} samples per '
                  f'{d["duration_hr"]:.0f} h)'],
            ],
        ),

        html.Div('Current model', style={
            'fontSize': '12px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '14px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        dcc.Markdown(
            "The simulated water-current profile has **three components** "
            "(plus a bottom boundary-layer correction):",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        dcc.Markdown(
            "- **Tidal current** — barotropic M2 tide, oscillates at the "
            "12.4 h lunar period; same magnitude at every depth.\n"
            "- **Wind-driven surface current** — strongest at the "
            "surface, decays exponentially with depth.\n"
            "- **Bottom boundary layer (BBL)** — seabed friction reduces "
            "current linearly to zero in the bottom few metres.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'u_{\text{tide}}(t) \;=\; A_{M_2}\,\cos\!\left(\dfrac{2\pi t}{T_{M_2}}\right)'
        ),
        latex_equation(
            r'u_{\text{wind}}(z) \;=\; u_{0}\,\exp\!\left(-\dfrac{|z|}{L_{e}}\right)'
        ),
        latex_equation(
            rf"f_{{\text{{BBL}}}}(z) \;=\; \min\!\left(1,\; "
            rf"\dfrac{{h - |z|}}{{{d['bbl_thickness']:.0f}\,\text{{m}}}}\right)"
        ),
        latex_equation(
            r'u(t,\,z) \;=\; \big[\,u_{\text{tide}}(t) \;+\; u_{\text{wind}}(z)\,\big] \cdot f_{\text{BBL}}(z)'
        ),
        styled_table(
            headers=['Component', 'Parameter', 'Value'],
            rows=[
                ['M2 tidal amplitude',
                  'A_M2',
                  f'±{d["M2_amplitude"]:.2f} m/s'],
                ['M2 tidal period',
                  'T_M2',
                  f'{d["M2_period_hr"]:.1f} h'],
                ['Wind surface current',
                  'u_0',
                  f'{d["wind_surface"]:.2f} m/s at surface'],
                ['Wind e-folding scale',
                  'L_e',
                  f'{d["wind_e_fold"]:.1f} m'],
                ['Bottom boundary layer thickness',
                  '(BBL)',
                  f'{d["bbl_thickness"]:.0f} m above seabed  '
                  f'(linear taper to 0)'],
            ],
        ),

        html.Div('Free-drifting platform dynamics  —  why we need GPS fusion', style={
            'fontSize': '12px', 'fontWeight': '600',
            'color': COLORS['text_dim'], 'margin': '14px 0 6px 0',
            'textTransform': 'uppercase', 'letterSpacing': '0.05em',
        }),
        dcc.Markdown(
            "Here's the catch with a hull-mounted ADCP on a **free-"
            "drifting** float: the ADCP measures **water velocity "
            "relative to the instrument**, not absolute earth-frame "
            "velocity. So the raw ADCP reading is biased by the "
            "platform's own motion:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'u_{\text{raw}}(t, z) \;=\; u(t, z) \;-\; u_{\text{device}}(t)'
        ),
        dcc.Markdown(
            "We measure $u_{\\text{device}}$ separately with **GPS** "
            "(position differencing → velocity), then add it back to "
            "recover the absolute current:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'u_{\text{corrected}}(t, z) \;=\; u_{\text{raw}}(t, z) \;+\; u_{\text{device}}^{\text{GPS}}(t)'
        ),
        dcc.Markdown(
            "The float itself drifts because it's pulled by the surface "
            "current with a small extra wind-drag offset:",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '10px 0 0 0'},
        ),
        latex_equation(
            r'u_{\text{device}}(t) \;=\; \alpha_{\text{coup}} \cdot u(t,\,z=0) \;+\; u_{\text{drag}}'
        ),
        styled_table(
            headers=['Parameter', 'Symbol', 'Value'],
            rows=[
                ['Surface coupling',
                  'α_coup',
                  f'{d["surface_coupling"]:.2f}  '
                  f'({d["surface_coupling"]*100:.0f}% of surface current)'],
                ['Extra wind drag',
                  'u_drag',
                  f'+{d["wind_drag"]:.2f} m/s'],
                ['GPS velocity noise (1σ)',
                  '—',
                  f'{d["gps_noise_cm_s"]:.1f} cm/s'],
                ['Fusion residual error  (noise floor)',
                  '—',
                  html.Span(f'{d["fusion_residual_cm_s"]:.2f} cm/s',
                             style={'fontWeight': '700',
                                     'color': COLORS['accent_2']})],
            ],
            footer_note='Fusion residual error = standard deviation of '
                        '(corrected velocity − true velocity), which sets '
                        'the noise floor of the corrected ADCP measurement.',
        ),

        section_header('The 4-step fusion pipeline'),

        # Panel 1 — caption ABOVE, then plot, then description
        html.Div('Step 1 — ground-truth absolute current  u(t, z)', style={
            'fontSize': '14px', 'fontWeight': '700',
            'color': COLORS['text'], 'margin': '20px 0 4px 0',
        }),
        dcc.Markdown(
            "The absolute water current we would *like* to measure — the "
            "sum of the three current-model components above:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'u(t,\,z) \;=\; \big[\,u_{\text{tide}}(t) \;+\; u_{\text{wind}}(z)\,\big] \cdot f_{\text{BBL}}(z)'
        ),
        dcc.Graph(figure=fig_p1, config={'displayModeBar': False}),
        dcc.Markdown(
            "Known here because we generated it. In real life this is what "
            "we're trying to recover.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # Panel 2 — raw ADCP
        html.Div('Step 2 — raw ADCP measurement  u_raw(t, z)', style={
            'fontSize': '14px', 'fontWeight': '700',
            'color': COLORS['text'], 'margin': '24px 0 4px 0',
        }),
        dcc.Markdown(
            "What the instrument actually reports — water velocity "
            "relative to the (drifting) hull:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'u_{\text{raw}}(t, z) \;=\; u(t, z) \;-\; u_{\text{device}}(t) \;+\; \text{ADCP noise}'
        ),
        dcc.Graph(figure=fig_p2, config={'displayModeBar': False}),
        dcc.Markdown(
            "Compared to Panel 1 the colours are uniformly shifted — "
            "this is the platform's own velocity contaminating every "
            "depth bin.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # Panel 3 — GPS device velocity
        html.Div('Step 3 — GPS-measured platform velocity  u_device(t)', style={
            'fontSize': '14px', 'fontWeight': '700',
            'color': COLORS['text'], 'margin': '24px 0 4px 0',
        }),
        dcc.Markdown(
            "Float velocity from GPS position differencing — a single "
            "time series, the same value applied at every depth bin "
            "(the whole hull moves together):",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'u_{\text{device}}^{\text{GPS}}(t) \;=\; u_{\text{device}}(t) \;+\; \text{GPS velocity noise}'
        ),
        dcc.Graph(figure=fig_p3, config={'displayModeBar': False}),
        dcc.Markdown(
            f"GPS noise floor is **{d['gps_noise_cm_s']:.1f} cm/s** — "
            f"small compared to the {d['M2_amplitude']*100:.0f} cm/s "
            f"tidal signal.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),

        # Panel 4 — corrected
        html.Div('Step 4 — fused output  u_corrected(t, z)', style={
            'fontSize': '14px', 'fontWeight': '700',
            'color': COLORS['text'], 'margin': '24px 0 4px 0',
        }),
        dcc.Markdown(
            "Add the GPS device velocity back to each raw bin to "
            "recover the absolute current:",
            mathjax=True,
            style={'fontSize': '13.5px', 'lineHeight': '1.65'},
        ),
        latex_equation(
            r'u_{\text{corrected}}(t, z) \;=\; u_{\text{raw}}(t, z) \;+\; u_{\text{device}}^{\text{GPS}}(t)'
        ),
        dcc.Graph(figure=fig_p4, config={'displayModeBar': False}),
        dcc.Markdown(
            f"Panel 4 should match Panel 1. Residual standard deviation "
            f"is **{d['fusion_residual_cm_s']:.2f} cm/s** — the noise "
            f"floor of the corrected current measurement, set by GPS "
            f"velocity noise + ADCP sensor noise.",
            style={'fontSize': '13.5px', 'lineHeight': '1.65',
                    'margin': '4px 0 18px 0'},
        ),
    ], style=STYLES['plots'])

    return html.Div([side, plots], style=STYLES['two_col'])


# ========================================================================
# APP LAYOUT
# ========================================================================
app = dash.Dash(
    __name__,
    title='WEC Sensor Dashboard',
    external_scripts=[
        'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js',
    ],
)
server = app.server   # for gunicorn deployment

TABS = [
    {'label': 'Overview',     'value': 'tab-0'},
    {'label': 'Motion',       'value': 'tab-1'},
    {'label': 'Strain & Fatigue', 'value': 'tab-2'},
    {'label': 'Resonance',    'value': 'tab-3'},
    {'label': 'Pressure',     'value': 'tab-4'},
    {'label': 'ADCP',         'value': 'tab-5'},
]

app.layout = html.Div([
    html.Div([
        html.H1('Wave-Energy Converter Sensor Dashboard',
                 style=STYLES['title']),
        html.Div('Physics-based simulation of float sensors with anomaly '
                 'detection, structural-health monitoring, and fatigue-life '
                 'estimation — a fleet-engineering prototype.',
                 style=STYLES['subtitle']),
    ], style=STYLES['header']),

    html.Div([
        dcc.Tabs(id='tabs', value='tab-0',
                 children=[
                    dcc.Tab(label=t['label'], value=t['value'],
                             style=STYLES['tab'], selected_style=STYLES['tab_selected'])
                    for t in TABS
                 ]),
    ], style=STYLES['tabs_container']),

    html.Div(id='tab-content', style=STYLES['content']),
], style=STYLES['app'])


@callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab_value):
    return {
        'tab-0': tab_overview,
        'tab-1': tab_motion,
        'tab-2': tab_fatigue,
        'tab-3': tab_strain,
        'tab-4': tab_pressure,
        'tab-5': tab_adcp,
    }.get(tab_value, tab_overview)()


@callback(
    Output('fatigue-output', 'children'),
    [Input('fatigue-scenario',    'value'),
     Input('fat-sigma-lim-air',   'value'),
     Input('fat-env-factor',      'value'),
     Input('fat-yield',           'value'),
     Input('fat-UTS',             'value')],
)
def update_fatigue_output(scenario, sigma_lim_air, env_factor, yield_str, UTS):
    """Re-run the fatigue simulation with new material parameters and
    rebuild the entire content area (sidebar + scenario figures +
    all-scenarios bar chart + summary table). Cost ~700 ms because of
    the 200-realisation Monte Carlo × 6 scenarios."""
    f = make_fatigue_data(
        sigma_limit_air=float(sigma_lim_air),
        env_factor=float(env_factor),
        yield_str=float(yield_str),
        UTS=float(UTS),
    )
    return _tab_fatigue_body(f, scenario)


# ----- Resonance tab — 4 sliders drive the entire Resonance body -----
@callback(
    Output('resonance-output', 'children'),
    [Input('res-zeta',     'value'),
     Input('res-m-tons',   'value'),
     Input('res-k-kNpm',   'value'),
     Input('res-Fdemo-kN', 'value')],
)
def update_resonance_output(zeta, m_tons, k_kNpm, F_demo_kN):
    """Re-run the strain simulation (rainflow / FFT / Monte Carlo all
    depend on m, k, c) and rebuild the Resonance tab body. Cost: ~0.3 s
    on a modern laptop because the FFT / IFFT path is fast.
    """
    d = make_strain_data(
        m=float(m_tons) * 1000.0,        # tons → kg
        k=float(k_kNpm) * 1000.0,        # kN/m → N/m
        zeta=float(zeta),
        F_demo_N=float(F_demo_kN) * 1000.0,   # kN → N
    )
    return _tab_strain_body(d)


# ----- Pressure tab — 2 sliders drive PTO-pressure simulation + prose -----
@callback(
    Output('pressure-output', 'children'),
    [Input('pre-pto-base',  'value'),
     Input('pre-pto-swing', 'value')],
)
def update_pressure_output(pto_base, pto_swing):
    """Re-run the pressure simulation with new PTO baseline / swing and
    rebuild the body. The hull / cabin / depth channels do not depend
    on these parameters, but it's simplest to re-run the whole thing."""
    d = make_pressure_data(
        pto_baseline_bar=float(pto_base),
        pto_swing_bar=float(pto_swing),
    )
    return _tab_pressure_body(d)


# ----- Motion tab — 14 sliders drive the full motion simulation -----
@callback(
    Output('motion-output', 'children'),
    [Input('mot-A',       'value'),
     Input('mot-T',       'value'),
     Input('mot-theta',   'value'),
     Input('mot-RAOh',    'value'),
     Input('mot-RAOs',    'value'),
     Input('mot-RAOw',    'value'),
     Input('mot-PhH',     'value'),
     Input('mot-PhS',     'value'),
     Input('mot-PhW',     'value'),
     Input('mot-rollA',   'value'),
     Input('mot-pitchA',  'value'),
     Input('mot-rollPh',  'value'),
     Input('mot-pitchPh', 'value'),
     Input('mot-yaw',     'value')],
)
def update_motion_output(A_wave, T_wave, theta,
                          RAO_h, RAO_s, RAO_w,
                          ph_h, ph_s, ph_w,
                          rollA, pitchA, rollPh, pitchPh, yaw_rate):
    """Re-run the full motion simulation including IMU fusion. Cost
    ~300-400 ms because of the imufusion filter on 3600 samples."""
    d = make_motion_data(
        A_wave=float(A_wave),
        T_wave=float(T_wave),
        wave_direction_deg=float(theta),
        RAO_heave=float(RAO_h),
        RAO_surge=float(RAO_s),
        RAO_sway=float(RAO_w),
        phase_heave=float(ph_h),
        phase_surge=float(ph_s),
        phase_sway=float(ph_w),
        roll_amp_deg=float(rollA),
        pitch_amp_deg_imu=float(pitchA),
        roll_phase=float(rollPh),
        pitch_phase=float(pitchPh),
        yaw_drift_rate=float(yaw_rate),
    )
    return _tab_motion_body(d)


# ----- ADCP tab — 5 sliders drive the simulated current profile -----
@callback(
    Output('adcp-output', 'children'),
    [Input('adcp-M2',  'value'),
     Input('adcp-u0',  'value'),
     Input('adcp-Le',  'value'),
     Input('adcp-bbl', 'value'),
     Input('adcp-sc',  'value')],
)
def update_adcp_output(M2, u0, Le, bbl, sc):
    """Re-run the ADCP current simulation with new wave-field + coupling
    parameters and rebuild the body. The full pipeline (u_true → ADCP
    raw → GPS-fusion → corrected → residual) recomputes. Cost ~150 ms."""
    d = make_adcp_data(
        M2_amplitude=float(M2),
        wind_surface=float(u0),
        e_fold=float(Le),
        bbl_thickness=float(bbl),
        surface_coupling=float(sc),
    )
    return _tab_adcp_body(d)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)
