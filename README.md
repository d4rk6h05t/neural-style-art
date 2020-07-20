[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![GitHub release](https://img.shields.io/badge/release-v1.0.0-green)](https://github.com/devMichani/nano-db-manager.git)

# neural-style-art
Neural style transfer
For this moment use Pytho2.7
# Packages level OS
 - __python2.7__
 - __pip__
 - __virtualenv__
 - __tk__
# Pip packages
 - __jupyterlab__
 - __matplotlib__
 - __numpy__
 - __pandas__
 - __Pyllow__
 - __tensorflow__
 - __tensorflow-gpu__
 - __keras__
 
Works Optimizer Adam from tensorflow Again Xd

```python
# works Adam YEA!
best, best_loss = run_style_transfer(content_path, style_path, num_iterations=5)
```


![png](neural_transfer_style_files/neural_transfer_style_21_0.png)



```python
Image.fromarray(best)
```




![png](neural_transfer_style_files/neural_transfer_style_22_0.png)




```python
def show_results(best_img, content_path, style_path, show_large_final=True):
    plt.figure(figsize=(10, 5))
    content = load_img(content_path) 
    style = load_img(style_path)
    
    plt.subplot(1, 2, 1)
    imshow(content, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style, 'Style Image')

    if show_large_final: 
        plt.figure(figsize=(10, 10))

    plt.imshow(best_img)
    plt.title('Output Image')
    plt.show()
```


```python
show_results(best, content_path, style_path)
```


![png](neural_transfer_style_files/neural_transfer_style_24_0.png)


![png](neural_transfer_style_files/neural_transfer_style_24_1.png)
