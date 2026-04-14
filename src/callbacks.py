import matplotlib.pyplot as plt
import keras
import os

class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, dataset, results_dir='results', epoch_interval=40):
        self.dataset = dataset.shuffle(
            buffer_size=2048
        )
        self.epoch_interval = epoch_interval
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    def display(self, display_list, extra_title='', save_path=None):
        plt.figure(figsize=(15, 15))
        title = ['Input Image', 'True Mask', 'Predicted Mask']

        if len(display_list) > len(title):
            title.append(extra_title)

        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(display_list[i], cmap='gray')
            plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        plt.close()
        
    def create_mask(self, pred_mask):
        pred_mask = (pred_mask > 0.5).astype("int32")
        return pred_mask[0]
    
    def show_predictions(self, dataset, num=1, epoch=None):
        for image, mask in dataset.take(num):
            pred_mask = self.model.predict(image)
            
            # Save visualization if epoch is provided
            save_path = None
            if epoch is not None:
                save_path = os.path.join(self.results_dir, f'prediction_epoch_{epoch+1}.png')
            
            self.display([image[0], mask[0], self.create_mask(pred_mask)], save_path=save_path)
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch and epoch % self.epoch_interval == 0:
            self.show_predictions(self.dataset, epoch=epoch)
            print ('\nSample Prediction after epoch {}\n'.format(epoch+1))
