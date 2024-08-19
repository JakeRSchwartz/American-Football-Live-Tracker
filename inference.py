from ultralytics import YOLO

model = YOLO('models/best-3.pt')
results = model.predict('NFL_Clips/clip1.mov', save = True)
print(results[0])
print('===================== ' * 2)
for box in results[0].boxes:
    print(box)