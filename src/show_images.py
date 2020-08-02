tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))
 
Here are the content and style images we will use: 
plt.figure(figsize=(10,10))

content = load_img(content_path).astype('uint8')
style = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.show()