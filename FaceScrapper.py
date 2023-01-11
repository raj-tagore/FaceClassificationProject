import cv2

for i in range(1, 1000):
    if (i < 10):
        img = cv2.imread(r"D:\Padhai\img_align_celeba\00000" + str(i) + ".jpg")
    elif (i > 10 and i < 100):
        img = cv2.imread(r"D:\Padhai\img_align_celeba\0000" + str(i) + ".jpg")
    elif (i > 100 and i < 1000):
        img = cv2.imread(r'D:\Padhai\img_align_celeba\000' + str(i) + ".jpg")
    gImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    FaceCascade = cv2.CascadeClassifier(cv2.data.haarcascades
                                        + 'haarcascade_frontalface_default.xml')
    faces = FaceCascade.detectMultiScale(gImg, scaleFactor=1.3,
                                         minNeighbors=3,
                                         minSize=(30, 30))
    print("found faces:", format(len(faces)))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = img[y:y + h, x:x + w]
        print("object found: saving.")
        cv2.imwrite(r'D:\Padhai\facedata2\faces' + str(i) + '.jpg', roi_color)




