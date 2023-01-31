from tkinter import *
from tkinter import ttk, filedialog
import os
import cv2

win = Tk()

win.title("Resim Analizi")
win.geometry("700x350")

def open_file():
	file = filedialog.askopenfile(mode='r', filetypes=[('Resim Dosyalar', '*.*')])

	if file:
		filepath = os.path.abspath(file.name)
		imagePath = str(filepath)

		image = cv2.imread(imagePath)
		image = cv2.resize(image, (640, 480))
		h = image.shape[0]
		w = image.shape[1]

		weights = "ssd_mobilenet/frozen_inference_graph.pb"
		model = "ssd_mobilenet/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"

		net = cv2.dnn.readNetFromTensorflow(weights, model)

		class_names = []

		with open("ssd_mobilenet/coco_names.txt", "r", encoding="utf-8") as f:
			class_names = f.read().strip().split("\n")

		blob = cv2.dnn.blobFromImage(
			image, 1.0 / 127.5, (320, 320), [127.5, 127.5, 127.5])

		net.setInput(blob)
		output = net.forward()

		for detection in output[0, 0, :, :]:
			probability = detection[2]

			if probability < 0.5:
				continue

			box = [int(a * b) for a, b in zip(detection[3:7], [w, h, w, h])]
			box = tuple(box)

			cv2.rectangle(image, box[:2], box[2:], (0, 255, 0), thickness=2)

			class_id = int(detection[1])
			class_name = class_names[class_id - 1]

			label = f"{class_name.upper()} {probability * 100:.2f}%"
			cv2.putText(image, label, (box[0], box[1] + 15),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

		cv2.imshow('Resim', image)
		cv2.waitKey()

label = Label(win, text="Resime göz atmak için aşağıdaki butona tıklayın.", font=('Georgia 13'))
label.pack(pady=10)

ttk.Button(win, text="Resim Seç", command=open_file).pack(pady=20)

win.mainloop()
