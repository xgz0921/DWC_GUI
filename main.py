import tkinter as tk
from tkinter import ttk, filedialog
from PIL import ImageTk
from functions import *
from IIDAO_env import IIDAO_Env

class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Slider Demo")

        # File selection
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.mat *.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if not file_path:
            root.destroy()
            return

        # Load env
        self.env = IIDAO_Env(single_mat_load(file_path), 14)


        # --- Frames for layout ---
        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        # left: sliders
        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(side="left", fill="y", padx=10, pady=10)

        reset_btn = ttk.Button(slider_frame, text="Reset Sliders", command=self.reset_sliders)
        reset_btn.pack(pady=10)

        # right: images
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)


        # --- Canvas for images + text ---
        self.canvas_left = tk.Canvas(image_frame,width=500,height=500)
        self.canvas_left.grid(row=0, column=0,columnspan=2, padx=5, pady=5)

        self.canvas_right = tk.Canvas(image_frame,width=300,height=300)
        self.canvas_right.grid(row=1, column=0, columnspan=1,padx=5, pady=5)

        self.canvas_obs = tk.Canvas(image_frame, width=300, height=300)
        self.canvas_obs.grid(row=1, column=1, columnspan=1, padx=5, pady=5)

        # initial left image
        self.tk_img_left = ImageTk.PhotoImage(self.env.img_intensity_display)
        self.left_img_obj = self.canvas_left.create_image(0, 0, anchor="nw", image=self.tk_img_left)
        self.left_text_obj = self.canvas_left.create_text(
            10, 10, anchor="nw", text="Left Image", fill="red", font=("Arial", 14, "bold")
        )
        #self.left_title_obj = self.canvas_left.

        # initial right image
        self.tk_img_right = ImageTk.PhotoImage(self.env.vr.psf)
        self.right_img_obj = self.canvas_right.create_image(0, 0, anchor="nw", image=self.tk_img_right)
        self.right_text_obj = self.canvas_right.create_text(
            150, 10, anchor="nw", text="PSF", fill="blue", font=("Arial", 14, "bold")
        )

        obs = self.env.get_observation(10, 3)
        self.obs_display = self.env.convert_image_to_PIL(obs)
        self.tk_img_obs = ImageTk.PhotoImage(self.obs_display)
        self.obs_img_obj = self.canvas_obs.create_image(0, 0, anchor="nw", image=self.tk_img_obs)
        self.obs_text_obj = self.canvas_obs.create_text(
            130, 10, anchor="nw", text="Feature Map", fill="red", font=("Arial", 14, "bold")
        )
        # --- Sliders ---
        co_range = 3
        self.slider_defs = [
            ("X-Tilt", -co_range, co_range, 0),
            ("Y-Tilt", -co_range, co_range, 0),
            ("Defocus", -co_range, co_range, 0),
            ("Obl. Ast.", -co_range, co_range, 0),
            ("Ver. Ast.", -co_range, co_range, 0),
            ("V. Coma", -co_range, co_range, 0),
            ("H. Coma", -co_range, co_range, 0),
            ("Ver. Trefoil", -co_range, co_range, 0),
            ("Obl. Trefoil", -co_range, co_range, 0),
            ("Spherical", -co_range, co_range, 0),
            ("Ver.2 Ast.", -co_range, co_range, 0),
            ("Obl.2 Ast.", -co_range, co_range, 0),
            ("Ver. Q.foil.", -co_range, co_range, 0),
            ("Obl. Q.foil.", -co_range, co_range, 0),
        ]

        self.slider_vars_list = []
        for (name, minval, maxval, init) in self.slider_defs:
            row_frame = ttk.Frame(slider_frame)
            row_frame.pack(fill="x", pady=2)

            # label on the left
            ttk.Label(row_frame, text=name, width=12).grid(row=0, column=0, sticky="w")

            var = tk.DoubleVar(value=init)
            slider = ttk.Scale(
                row_frame, from_=minval, to=maxval,
                orient="horizontal", variable=var,
                command=self.update_image,
                length=150
            )
            slider.grid(row=0, column=1, padx=5)

            # live value on the right
            value_label = ttk.Label(row_frame, text=f"{init:.2f}", width=6)
            value_label.grid(row=0, column=2, sticky="e")

            # save both var and its value label for updates
            self.slider_vars_list.append((var, value_label))

    def update_image(self, event=None):
        # collect slider values
        cr = []
        for var, value_label in self.slider_vars_list:
            value_label.config(text=f"{var.get():.2f}")
            cr.append(var.get())
        # update env
        self.env.crSet(cr)
        sm = self.env.sharpness_metric
        em = self.env.entropy_metric
        # --- Left image ---
        img_left = self.env.img_intensity_display.copy()
        self.tk_img_left = ImageTk.PhotoImage(img_left)
        self.canvas_left.itemconfig(self.left_img_obj, image=self.tk_img_left)
        self.canvas_left.itemconfig(self.left_text_obj, text=f'Sharpness: {sm}\nEntropy: {em}')
        # --- Right image ---
        img_right = self.env.vr.psf.copy()
        self.tk_img_right = ImageTk.PhotoImage(img_right)
        self.canvas_right.itemconfig(self.right_img_obj, image=self.tk_img_right)

        #%%  obs
        obs = self.env.get_observation(10,3)
        self.obs_display = self.env.convert_image_to_PIL(obs)
        self.tk_img_obs = ImageTk.PhotoImage(self.obs_display)
        self.canvas_obs.itemconfig(self.obs_img_obj, image=self.tk_img_obs)

    def reset_sliders(self):
        for (var,_), (_, _, _, init) in zip(self.slider_vars_list, self.slider_defs):
            var.set(init)
        self.update_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
