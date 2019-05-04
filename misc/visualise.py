import numpy as np
import sys
import matplotlib.pyplot as plt 
import os
import cv2

if len(sys.argv) != 2:
	print('Error in the length of the sys')
	exit(-1)

if sys.argv[1] == 'All':

	all_ = os.listdir('../Temporary')
	for all_i in all_:
		x = np.load('../Temporary/'+all_i)
		center = x[:, :, 3]<x[:, :, 4]
		center = center.astype(np.float32).astype(np.uint8)*255

		image, contours, hierarchy =   cv2.findContours(center ,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

		center_co = []
		cont = []
		full_cont = []
		print(np.array(contours).shape)

		for co_ in contours:

			left, right = np.min(co_[:, 0, 0]), np.max(co_[:, 0, 0])
			up, down = np.min(co_[:, 0, 1]), np.max(co_[:, 0, 1])
			center_co.append([up, down, left, right, (up + down)//2, (right + left)//2])
			cont.append(np.array([[left, up], [right, up], [right, down], [left, down]]).reshape([4, 1, 2]))
			left_con, right_con = (right + left)//2 - (right - left)//2*5, (right + left)//2 + (right - left)//2*5
			up_con, down_con = (up + down)//2 - (down - up)//2*5, (up + down)//2 + (down - up)//2*5
			full_cont.append(np.array([[left_con, up_con], [right_con, up_con], [right_con, down_con], [left_con, down_con]]).reshape([4, 1, 2]))


		print(np.array(cont).shape)

		plt.imshow(cv2.drawContours((x[:, :, 0:3]*255).astype(np.uint8) ,full_cont,-1,(0,255,0),3))
		plt.show()


		# plt.imsave('first.png', x[:, :, :3])
		# plt.imsave('second.png', x[:, :, 3])
		# plt.imsave('third.png', x[:, :, 4])
		# plt.imsave('fourth.png', x[:, :, 5])
		# plt.imsave('fifth.png', x[:, :, 6])
		# plt.imsave('sixth.png', x[:, :, 7])
		
		# exit(-1)
		# plt.imshow(x[:, :, :3])
		# plt.pause(0.4)

		# plt.imshow(x[:, :, 3])
		# plt.pause(0.4)

		# plt.imshow(x[:, :, 4])
		# plt.pause(0.4)

		# plt.imshow(x[:, :, 5])
		# plt.pause(0.4)

		# plt.imshow(x[:, :, 6])
		# plt.pause(0.2)

		# plt.imshow(x[:, :, 7])
		# plt.pause(0.2)

		plt.clf()

else:
	
	if not os.path.exists('../Temporary/'+sys.argv[1]):
		print('Error, path does not exist')
		exit(-1)

	x = np.load('../Temporary/'+sys.argv[1])

	plt.imshow(x[:, :, :3])
	plt.show()

	plt.imshow(x[:, :, 3])
	plt.show()

	plt.imshow(x[:, :, 4])
	plt.show()

	plt.imshow(x[:, :, 5])
	plt.show()