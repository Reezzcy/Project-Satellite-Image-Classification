import tkinter as tk
from tkinter import filedialog, Label
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import cv2
from skimage.feature import local_binary_pattern
import folium
from io import BytesIO
import webbrowser

# Load model yang sudah dilatih
model = tf.keras.models.load_model('model_satellite.h5')

# Parameter untuk Local Binary Pattern (LBP)
RADIUS = 3
N_POINTS = 8 * RADIUS

def preprocess_image(image_path):
    # Membaca citra dalam mode grayscale
    img = cv2.imread(image_path, 0)

    # Menerapkan gaussian blur pada citra
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)

    # Ekstraksi fitur pada citra dengan LBP
    lbp_img = local_binary_pattern(blurred_img, N_POINTS, RADIUS, 'uniform')

    # Normalisasi gambar yang sudah diekstrak fiturnya dengan jarak [0, 255]
    lbp_normalized = cv2.normalize(lbp_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Resize citra menjadi 100x100 pixels
    img_resized = cv2.resize(lbp_normalized, (100, 100))

    # Normalisasi nilai pixel citra
    img_norm = img_resized / 255.0

    # Reshape citra untuk menyamakan citra seperti bentuk input pada model
    img_final = img_norm.reshape(1, 100, 100, 1)

    return img_final

def predict_image(image_path):
    # Preprocess citra
    processed_image = preprocess_image(image_path)
    # Prediksi kelas dari citra yang di input menggunakan model yang sudah dilatih
    prediction = model.predict(processed_image)
    index = np.argmax(prediction)

    # Definisikan nama-nama kelas    
    class_names = ['Cloudy', 'Desert', 'Green Area', 'Water']
    predict = class_names[index]
    return predict

class ImageClassifierApp:
    def __init__(self, root):
        # Inisialisasi window utama
        self.root = root
        self.root.title("Klasifikasi Citra Satelit")
        self.root.iconbitmap('LOGO UPNVJ.ico')
        
        # Label judul
        self.title_label = Label(root, text="Klasifikasi Citra Satelit", font=("Arial", 24))
        self.title_label.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')
        
        # Tombol untuk load citra
        self.load_button = tk.Button(root, text="Load File", padx=10, pady=5, command=self.upload_image)
        self.load_button.grid(row=1, column=0, padx=10, pady=10, sticky='nw')

        # Labels untuk menampilkan citra asli dan yang sudah diproses
        self.original_image_label = tk.Label(root)
        self.original_image_label.grid(row=1, column=1, rowspan=3, padx=10, pady=10, sticky='nsew')

        self.processed_image_label_img = tk.Label(root)
        self.processed_image_label_img.grid(row=1, column=2, rowspan=3, padx=10, pady=10, sticky='nsew')

        # Label untuk menampilkan kelas hasil prediksi
        self.prediction_label = tk.Label(root, text="Prediction: None", padx=10, pady=5)
        self.prediction_label.grid(row=4, column=0, columnspan=3, padx=10, pady=5, sticky='nsew')

        # Tombol untuk membuka peta
        self.map_button = tk.Button(root, text="Show Map", padx=10, pady=5, command=self.show_map)
        self.map_button.grid(row=5, column=0, columnspan=3, padx=10, pady=10, sticky='nsew')

        self.predicted_class = None

        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(2, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)
        root.grid_rowconfigure(3, weight=1)

    def upload_image(self):
        # Buka file untuk memilih file citra
        file_path = filedialog.askopenfilename()
        if file_path:
            # Tampilkan citra asli
            original_image = Image.open(file_path)
            original_image = original_image.resize((300, 300), Image.LANCZOS)
            original_image = ImageTk.PhotoImage(original_image)
            self.original_image_label.config(image=original_image)
            self.original_image_label.image = original_image

            # Proses citra input dan tampilkan
            processed_image = preprocess_image(file_path)
            processed_image = np.squeeze(processed_image) * 255.0
            processed_image = Image.fromarray(processed_image.astype(np.uint8))
            processed_image = processed_image.resize((300, 300), Image.LANCZOS)
            processed_image = ImageTk.PhotoImage(processed_image)
            self.processed_image_label_img.config(image=processed_image)
            self.processed_image_label_img.image = processed_image

            # Prediksi kelas dari citra input dan tampilkan hasil prediksi
            self.predicted_class = predict_image(file_path)
            self.prediction_label.config(text=f"Prediction: {self.predicted_class}")

    def show_map(self):
        # Koordinat Indonesia
        coordinates = [-0.789275, 113.921327]

        # Buat peta dengan Indonesia sebagi titik pusat
        m = folium.Map(location=coordinates, zoom_start=5)

        # Tanda/Pin berdasarkan hasil prediksi kelas
        if self.predicted_class == 'Desert':
            folium.Marker(
                location=[-7.92967, 112.96586],
                popup="Padang Pasir Bromo",
                icon=folium.Icon(color='red', icon='info-sign')
            ).add_to(m)
        elif self.predicted_class == 'Green Area':
            folium.Marker(
                location=[-6.8573768, 107.6286693],
                popup="Taman Hutan Raya Ir.H.Djuanda",
                icon=folium.Icon(color='green', icon='info-sign')
            ).add_to(m)
        elif self.predicted_class == 'Water':
            folium.Marker(
                location=[-5.137524, 112.1586481],
                popup="Laut Jawa",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(m)

        # Simpan peta dalam format html
        map_data = BytesIO()
        m.save(map_data, close_file=False)

        # Buka peta pada browser
        map_html = map_data.getvalue().decode()
        map_file_path = "map.html"
        with open(map_file_path, 'w') as map_file:
            map_file.write(map_html)

        webbrowser.open(map_file_path)

if __name__ == "__main__":
    # Loop untuk menjalankan aplikasi
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()