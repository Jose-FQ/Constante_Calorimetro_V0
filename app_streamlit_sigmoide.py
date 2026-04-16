from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
try:
    import streamlit as st
except Exception:
    st = None
from scipy.optimize import curve_fit


APP_TITLE = 'Ajuste sigmoide de temperatura vs tiempo'
APP_SUBTITLE = 'Laboratorio de Termodinamica - 2026'
APP_NOTE = (
    'Ajuste no lineal con una funcion sigmoide para sistemas con dos mesetas y una transicion brusca.'
)


class FitResult:
    def __init__(self, params, errors, y_pred, sse, rmse, r2, aic):
        self.params = params
        self.errors = errors
        self.y_pred = y_pred
        self.sse = sse
        self.rmse = rmse
        self.r2 = r2
        self.aic = aic


DEFAULT_TEXT = '''# tiempo(s) temperatura(C)
0 21.4
15 21.8
30 23.0
45 22.2
60 22.7
75 22.5
90 21.7
105 22.5
120 21.9
135 21.9
150 22.2
165 21.1
180 23.2
195 22.5
210 25.7
225 30.7
240 42.4
255 51.5
270 57.4
285 59.9
300 58.7
315 61.0
330 61.6
345 61.3
360 61.0
375 60.4
390 61.4
405 61.3
420 60.8
435 60.5
450 60.2
465 59.6
480 61.4
495 59.0
510 60.2
525 60.6
540 61.1
555 61.7
570 61.2
585 60.3
600 61.0
'''


def sigmoid_model(t: np.ndarray, y1: float, y2: float, t_star: float, k: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return y1 + (y2 - y1) / (1.0 + np.exp(-k * (t - t_star)))


def parse_data_text(text: str) -> pd.DataFrame:
    if text is None or not text.strip():
        raise ValueError('No se proporcionaron datos.')

    df = pd.read_csv(
        io.StringIO(text),
        comment='#',
        sep=r'\s+|,|;',
        engine='python',
        header=None,
        names=['tiempo_s', 'temperatura_C'],
    ).dropna()

    if df.empty:
        raise ValueError('No fue posible leer datos validos desde el texto proporcionado.')

    df['tiempo_s'] = pd.to_numeric(df['tiempo_s'], errors='coerce')
    df['temperatura_C'] = pd.to_numeric(df['temperatura_C'], errors='coerce')
    df = df.dropna().sort_values('tiempo_s').reset_index(drop=True)

    if len(df) < 5:
        raise ValueError('Se requieren al menos 5 pares tiempo-temperatura para realizar el ajuste.')

    if df['tiempo_s'].duplicated().any():
        raise ValueError('Hay tiempos repetidos. Usa un solo valor de temperatura por tiempo.')

    return df


def parse_uploaded_file(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue().decode('utf-8', errors='ignore')
    return parse_data_text(raw)


def initial_guess(t: np.ndarray, y: np.ndarray) -> list[float]:
    y_low = float(np.percentile(y, 10))
    y_high = float(np.percentile(y, 90))
    y_mid = 0.5 * (y_low + y_high)
    idx_mid = int(np.argmin(np.abs(y - y_mid)))
    t_star0 = float(t[idx_mid])
    dt = float(np.median(np.diff(np.sort(t)))) if len(t) > 1 else 1.0
    k0 = max(1.0 / max(4.0 * dt, 1.0), 1e-5)
    return [y_low, y_high, t_star0, k0]


def fit_sigmoid(df: pd.DataFrame) -> FitResult:
    t = df['tiempo_s'].to_numpy(dtype=float)
    y = df['temperatura_C'].to_numpy(dtype=float)

    p0 = initial_guess(t, y)
    ymin, ymax = float(np.min(y)), float(np.max(y))
    dt = float(np.median(np.diff(np.sort(t)))) if len(t) > 1 else 1.0

    bounds = (
        [ymin - 20.0, ymin - 20.0, float(t.min()), 1e-8],
        [ymax + 20.0, ymax + 20.0, float(t.max()), 2.0 / max(dt, 1e-9)],
    )

    popt, pcov = curve_fit(sigmoid_model, t, y, p0=p0, bounds=bounds, maxfev=50000)
    perr = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
    y_pred = sigmoid_model(t, *popt)
    resid = y - y_pred
    sse = float(np.sum(resid ** 2))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float('nan')
    aic = float(len(y) * np.log(sse / len(y)) + 2 * len(popt)) if sse > 0 else float('-inf')

    return FitResult(
        params=popt,
        errors=perr,
        y_pred=y_pred,
        sse=sse,
        rmse=rmse,
        r2=r2,
        aic=aic,
    )


def fitted_equation_latex(params: np.ndarray) -> str:
    y1, y2, t_star, k = params
    return (
        rf"T(t) = {y1:.4f} + \frac{{{(y2 - y1):.4f}}}{{1 + e^{{-{k:.6f}(t - {t_star:.4f})}}}}"
    )


def parameter_table(result: FitResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'Parametro': ['y1', 'y2', 't_star', 'k'],
            'Valor': result.params,
            'Error': result.errors,
            'Unidad': ['degC', 'degC', 's', 's^-1'],
            'Interpretacion': [
                'Primera meseta',
                'Segunda meseta',
                'Tiempo caracteristico de transicion',
                'Rapidez de la transicion',
            ],
        }
    )


def summary_table(result: FitResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'Metrica': ['SSE', 'RMSE', 'R^2', 'AIC'],
            'Valor': [result.sse, result.rmse, result.r2, result.aic],
        }
    )


def make_plot(df: pd.DataFrame, result: FitResult):
    t = df['tiempo_s'].to_numpy(dtype=float)
    y = df['temperatura_C'].to_numpy(dtype=float)
    dense_t = np.linspace(float(t.min()), float(t.max()), 1500)
    dense_y = sigmoid_model(dense_t, *result.params)

    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    ax.scatter(t, y, s=34, label='Datos experimentales', zorder=3)
    ax.plot(dense_t, dense_y, linewidth=2.4, label='Ajuste sigmoide')
    ax.axhline(result.params[0], linestyle='--', linewidth=1.3, label=f'y1 = {result.params[0]:.3f} degC')
    ax.axhline(result.params[1], linestyle='--', linewidth=1.3, label=f'y2 = {result.params[1]:.3f} degC')
    ax.axvline(result.params[2], linestyle=':', linewidth=1.5, label=f't* = {result.params[2]:.3f} s')
    ax.set_title('Ajuste sigmoide de temperatura vs tiempo')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Temperatura (degC)')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    return fig


def build_report_pdf(df: pd.DataFrame, result: FitResult) -> bytes:
    params_df = parameter_table(result)
    metrics_df = summary_table(result)
    pdf_buffer = io.BytesIO()

    with PdfPages(pdf_buffer) as pdf:
        fig = make_plot(df, result)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8.2))
        ax.axis('off')
        ax.text(0.03, 0.96, APP_TITLE, fontsize=17, fontweight='bold', transform=ax.transAxes)
        ax.text(0.03, 0.92, APP_SUBTITLE, fontsize=11, transform=ax.transAxes)
        ax.text(0.03, 0.88, 'Modelo utilizado: funcion sigmoide', fontsize=11, transform=ax.transAxes)
        ax.text(0.03, 0.83, 'Funcion ajustada:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        ax.text(0.03, 0.77, '$' + fitted_equation_latex(result.params) + '$', fontsize=12, transform=ax.transAxes)

        p_rows = []
        for _, row in params_df.iterrows():
            p_rows.append([
                row['Parametro'],
                f"{row['Valor']:.6f}",
                f"{row['Error']:.6f}",
                row['Unidad'],
                row['Interpretacion'],
            ])
        tab = ax.table(
            cellText=p_rows,
            colLabels=['Parametro', 'Valor', 'Error', 'Unidad', 'Interpretacion'],
            cellLoc='center',
            bbox=[0.03, 0.40, 0.94, 0.28],
        )
        tab.auto_set_font_size(False)
        tab.set_fontsize(10)
        tab.scale(1, 1.25)

        m_rows = [[m, f'{v:.6f}'] for m, v in metrics_df.to_numpy()]
        tab2 = ax.table(
            cellText=m_rows,
            colLabels=['Metrica', 'Valor'],
            cellLoc='center',
            bbox=[0.03, 0.22, 0.42, 0.12],
        )
        tab2.auto_set_font_size(False)
        tab2.set_fontsize(10)
        tab2.scale(1, 1.25)

        y1, y2 = result.params[0], result.params[1]
        e1, e2 = result.errors[0], result.errors[1]
        ax.text(
            0.03,
            0.13,
            f'Primera meseta: {y1:.4f} +/- {e1:.4f} degC\\nSegunda meseta: {y2:.4f} +/- {e2:.4f} degC',
            fontsize=11,
            transform=ax.transAxes,
        )
        ax.text(
            0.03,
            0.06,
            'Los errores de estimacion se obtienen de la matriz de covarianza del ajuste no lineal.',
            fontsize=9.5,
            transform=ax.transAxes,
        )

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def build_csv_bytes(result: FitResult) -> bytes:
    out = io.StringIO()
    parameter_table(result).to_csv(out, index=False)
    return out.getvalue().encode('utf-8')


def render_header():
    if st is None:
        raise RuntimeError('Streamlit no esta instalado en este entorno.')
    st.set_page_config(page_title=APP_TITLE, layout='wide')
    st.markdown("""
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {background: #f7f9fc; border: 1px solid #e4e8ef; padding: 0.8rem; border-radius: 0.8rem;}
    .small-note {color: #5b6575; font-size: 0.95rem;}
    </style>
    """, unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption(APP_SUBTITLE)
    st.markdown(APP_NOTE)
    st.divider()


def input_panel() -> pd.DataFrame | None:
    st.subheader('1. Ingreso de datos')
    mode = st.radio(
        'Selecciona como quieres introducir los datos:',
        ['Cargar archivo', 'Escribir o pegar datos'],
        horizontal=True,
    )

    df = None
    if mode == 'Cargar archivo':
        uploaded = st.file_uploader(
            'Carga un archivo TXT, CSV o DAT con dos columnas: tiempo(s) y temperatura(degC).',
            type=['txt', 'csv', 'dat'],
        )
        if uploaded is not None:
            df = parse_uploaded_file(uploaded)
    else:
        text = st.text_area(
            'Pega aqui los datos. Puedes separarlos por espacios, tabulaciones, comas o punto y coma.',
            value=DEFAULT_TEXT,
            height=320,
        )
        if st.button('Leer datos escritos', use_container_width=True):
            df = parse_data_text(text)
            st.session_state['manual_df'] = df
        elif 'manual_df' in st.session_state:
            df = st.session_state['manual_df']

    if df is not None:
        st.success(f'Se cargaron {len(df)} datos correctamente.')
        st.markdown('**Vista previa y edicion de datos**')
        edited_df = st.data_editor(df, use_container_width=True, num_rows='dynamic', hide_index=True)
        edited_df['tiempo_s'] = pd.to_numeric(edited_df['tiempo_s'], errors='coerce')
        edited_df['temperatura_C'] = pd.to_numeric(edited_df['temperatura_C'], errors='coerce')
        edited_df = edited_df.dropna().sort_values('tiempo_s').reset_index(drop=True)
        if len(edited_df) < 5:
            raise ValueError('Despues de editar, se requieren al menos 5 datos validos.')
        return edited_df
    return df


def show_results(df: pd.DataFrame, result: FitResult):
    st.subheader('2. Resultados del ajuste')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('y1', f'{result.params[0]:.4f} degC', f'+/- {result.errors[0]:.4f}')
    col2.metric('y2', f'{result.params[1]:.4f} degC', f'+/- {result.errors[1]:.4f}')
    col3.metric('t*', f'{result.params[2]:.4f} s', f'+/- {result.errors[2]:.4f}')
    col4.metric('k', f'{result.params[3]:.6f} s^-1', f'+/- {result.errors[3]:.6f}')

    fig = make_plot(df, result)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown('**Funcion sigmoide ajustada**')
    st.latex(fitted_equation_latex(result.params))

    c1, c2 = st.columns([1.3, 1.0])
    with c1:
        st.markdown('**Parametros y errores de estimacion**')
        st.dataframe(parameter_table(result), use_container_width=True, hide_index=True)
    with c2:
        st.markdown('**Metricas del ajuste**')
        st.dataframe(summary_table(result), use_container_width=True, hide_index=True)

    st.info(
        f"Meseta 1 = {result.params[0]:.4f} +/- {result.errors[0]:.4f} degC | "
        f"Meseta 2 = {result.params[1]:.4f} +/- {result.errors[1]:.4f} degC"
    )


def main():
    render_header()

    st.sidebar.header('Opciones')
    st.sidebar.markdown(
        'Esta aplicacion ajusta unicamente la funcion sigmoide y permite generar un informe en PDF.'
    )

    try:
        df = input_panel()
    except Exception as exc:
        st.error(f'Error al leer los datos: {exc}')
        return

    if df is None:
        st.warning('Carga un archivo o pega los datos para continuar.')
        return

    st.subheader('3. Ejecutar ajuste')
    run_fit = st.button('Ajustar funcion sigmoide', type='primary', use_container_width=True)

    if not run_fit and 'fit_result' not in st.session_state:
        return

    if run_fit:
        try:
            result = fit_sigmoid(df)
            st.session_state['fit_result'] = result
            st.session_state['fit_df'] = df.copy()
        except Exception as exc:
            st.error(f'No fue posible realizar el ajuste: {exc}')
            return

    if 'fit_result' in st.session_state and 'fit_df' in st.session_state:
        result = st.session_state['fit_result']
        df_to_show = st.session_state['fit_df']
        show_results(df_to_show, result)

        st.subheader('4. Generacion del informe')
        pdf_bytes = build_report_pdf(df_to_show, result)
        csv_bytes = build_csv_bytes(result)
        st.download_button(
            'Descargar informe en PDF',
            data=pdf_bytes,
            file_name='reporte_ajuste_sigmoide.pdf',
            mime='application/pdf',
            use_container_width=True,
        )
        st.download_button(
            'Descargar parametros en CSV',
            data=csv_bytes,
            file_name='parametros_ajuste_sigmoide.csv',
            mime='text/csv',
            use_container_width=True,
        )


if __name__ == '__main__':
    main()
