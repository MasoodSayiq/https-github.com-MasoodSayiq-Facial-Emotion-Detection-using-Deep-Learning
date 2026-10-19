from data_preprocessing import load_and_preprocess
from model import build_model, get_callbacks

def train_model():
    train_ds, val_ds, test_ds = load_and_preprocess()
    
    model = build_model()
    callbacks = get_callbacks()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=callbacks
    )
    
    # Save the final model
    model.save("../data/Datasets/data/models/emotion_model.h5")
    print("Model training complete and saved!")
    
    return history

if __name__ == "__main__":
    train_model()
