from nicegui import ui
import VPIN.vpin_backend as vb
def content() -> None:
    """
    Main function for running the VPIN algorithms.

    This function sets up the user interface and handles user interactions.
    The user can choose to do the adjpin, pin and vpin methods on the data.

    Parameters:
        None

    Returns:
        None
    """
    with ui.card():
        ui.label("Upload Data")
        with ui.row():
            clean_data_upload = ui.upload(label="Upload clean data", on_upload=vb.handle_upload, auto_upload=True).classes('max-w-full')
            buckets_data_upload = ui.upload(label="Upload buckets data", on_upload=vb.handle_upload, auto_upload=True).classes('max-w-full')

    with ui.card():
        with ui.column():
            adjpin = ui.checkbox("AdjPIN", value=False)
            with ui.row():
                ui.label("Method")
                ap_method = ui.toggle(["ML", "ECM"])
                ap_initialsets = ui.number("Initialsets", value=20)
                ap_num_init = ui.number("Num_Init", value=20)

    with ui.card():
        with ui.column():
            pin = ui.checkbox("PIN", value=False)
            with ui.row():
                p_initialsets = ui.number("Initialsets", value=20)
                p_alpha = ui.number("alpha", value=0.3)
                p_delta = ui.number("delta", value=0.1)
            with ui.row():
                p_mu = ui.number("mu", value=800)
                p_epsilon_b = ui.number("epsilon_b", value=300)
                p_epsilon_s = ui.number("epsilon_s", value=200)
            with ui.row():
                ui.label("Factorization")
                p_factorization = ui.toggle(["E", "EHO", "LK"])

    with ui.card():
        with ui.column():
            vpin = ui.checkbox("VPIN", value=False)
            with ui.row():
                vp_time_bars = ui.number("Time Bars", value=60)
                vp_buckets = ui.number("Buckets", value=50)
                vp_sample_length = ui.number("Sample Length", value=50)
                vp_trading_hours = ui.number("Trading Hours", value=24)

    ui.button('Run Algorithm').on('click', lambda: vb.button_callback(adjpin, ap_method, ap_initialsets, ap_num_init, pin, p_initialsets, p_alpha, p_delta, p_mu, p_epsilon_b, p_epsilon_s, p_factorization, vpin, vp_time_bars, vp_buckets, vp_sample_length, vp_trading_hours)).classes(
                        'bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-full')

