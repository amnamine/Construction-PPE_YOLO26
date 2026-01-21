import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

# Load YOLO26 custom model for Construction-PPE
model = YOLO("construction.pt")  # Replace with your actual path to best.pt

# ---------------------- MAIN WINDOW ----------------------
window = tk.Tk()
window.title("Construction PPE Detector - YOLO26")
window.state("zoomed")  # Full screen
window.configure(bg="#2B2B2B")  # Dark gray background

# ---------------------- GLOBALS ----------------------
loaded_image = None
tk_img = None
panel = None

# ---------------------- FUNCTIONS ----------------------
def load_image():
    global loaded_image, tk_img, panel
    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        return
    loaded_image = Image.open(path).convert("RGB")

    display_img = loaded_image.copy()
    display_img.thumbnail((window.winfo_width() - 50, window.winfo_height() - 200))
    tk_img = ImageTk.PhotoImage(display_img)

    if panel is None:
        panel_frame = tk.Frame(window, bd=5, relief="ridge", bg="#444444")
        panel_frame.pack(pady=20, padx=20, fill="both", expand=True)
        panel = tk.Label(panel_frame, image=tk_img, bg="#444444")
        panel.pack(fill="both", expand=True)
    else:
        panel.configure(image=tk_img)
        panel.image = tk_img

def predict():
    global loaded_image, tk_img, panel
    if loaded_image is None:
        return
    # Run YOLO26 inference
    results = model(loaded_image)
    result = results[0]
    annotated = result.plot()  # Annotated image with boxes
    annotated = annotated[..., ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(annotated)
    pil_img.thumbnail((window.winfo_width() - 50, window.winfo_height() - 200))

    tk_img = ImageTk.PhotoImage(pil_img)
    panel.configure(image=tk_img)
    panel.image = tk_img

def reset():
    global loaded_image, tk_img, panel
    loaded_image = None
    tk_img = None
    if panel:
        panel.configure(image="")
        panel.image = None

# ---------------------- BUTTON STYLES ----------------------
button_font = ("Arial", 16, "bold")
btn_color = "#FF7F0F"  # Orange construction-themed
btn_hover = "#FFA500"

def on_enter(e):
    e.widget['bg'] = btn_hover

def on_leave(e):
    e.widget['bg'] = btn_color

# ---------------------- BUTTON FRAME ----------------------
btn_frame = tk.Frame(window, bg="#2B2B2B")
btn_frame.pack(pady=15)

btn_load = tk.Button(btn_frame, text="Load Image", font=button_font,
                     bg=btn_color, fg="white", width=15, command=load_image, bd=3, relief="raised")
btn_load.grid(row=0, column=0, padx=15)
btn_load.bind("<Enter>", on_enter)
btn_load.bind("<Leave>", on_leave)

btn_predict = tk.Button(btn_frame, text="Predict", font=button_font,
                        bg=btn_color, fg="white", width=15, command=predict, bd=3, relief="raised")
btn_predict.grid(row=0, column=1, padx=15)
btn_predict.bind("<Enter>", on_enter)
btn_predict.bind("<Leave>", on_leave)

btn_reset = tk.Button(btn_frame, text="Reset", font=button_font,
                      bg=btn_color, fg="white", width=15, command=reset, bd=3, relief="raised")
btn_reset.grid(row=0, column=2, padx=15)
btn_reset.bind("<Enter>", on_enter)
btn_reset.bind("<Leave>", on_leave)

# ---------------------- FOOTER LABEL ----------------------
footer = tk.Label(window, text="Construction PPE Detector - YOLO26 | Ultralytics",
                  bg="#2B2B2B", fg="white", font=("Arial", 12, "italic"))
footer.pack(side="bottom", pady=10)

# ---------------------- RUN WINDOW ----------------------
window.mainloop()
