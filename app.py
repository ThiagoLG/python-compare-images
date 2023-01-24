import cv2 
import numpy as np


print("teste")

original = cv2.imread('ronaldinho_1.jpg')
image_to_compare  = cv2.imread('ronaldinho_2.jpg')

if original.shape == image_to_compare.shape:
  print('imagens possuem mesmo tamanho e canais')
  difference = cv2.subtract(original, image_to_compare)
  b, g, r = cv2.split(difference)
  
  if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    print('Imagens completamente iguais')
  else: 
    print('imagens não são iguais')
    
sift = cv2.xfeatures2d.SIFT_create()

kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []

for m, n in matches:
  if m.distance < 0.9*n.distance:
    good_points.append(m)

number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)

print("Keypoints da primeira imagem: " + str(len(kp_1)))
print("Keypoints da segunda imagem: " + str(len(kp_2)))
print("GOOD Matches:", len(good_points))
print("How good it's the match: ", len(good_points) / number_keypoints * 100, "%")

