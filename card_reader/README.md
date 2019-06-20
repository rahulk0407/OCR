# IMPORTANT
crop the credit card image using the code i have uploaded for cropping the image or do it in a image editor, crop such that only 16 digit are in the image , resize the image to around 645 * 96. 
in test data upload data which is not same in the train data, for sake of simplicity i have uploaded the same data, 
you can use the card_reader code to create test data, write roi into any folder, let's name the folder example and declare i=0 and then 
<br />**cv2.imwrite("./example/"+str(i)+"./_1_"".jpg", roi_otsu)
i=i+1 do this in a loop[ for i in range(0,10):]**<br />
then using data_production.py use this images in example folder to create dataset.  
