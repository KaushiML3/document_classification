from app.src.vit_load import VITDocumentClassifier
from app.src.vgg16_load import VGGDocumentClassifier
from app.src.layout_loader import LayoutLMDocumentClassifier
from pathlib import Path



if __name__ == "__main__":
    try:
        #VIT model
        print("#########Test the VIT model #########")
        model_path=Path(r"artifacts\model\VIT_model\model.pth")
        mlb_path=Path(r"artifacts\model\VIT_model\mlb.joblib")
        vit=VITDocumentClassifier(model_path, mlb_path)
        result=vit.predict(Path(r"src_img\2070046307a.jpg"))
    
    except Exception as e:
        raise e
    

    try:
        print("#############Test the VGG model #############")
        model_path=Path(r"artifacts\model\vgg_model\model.keras")
        mlb_path=Path(r"artifacts\model\vgg_model\mlb.joblib")
        classifier = VGGDocumentClassifier(model_path,mlb_path)
        predicted_labels = classifier.predict(Path(r"src_img\2070046307a.jpg"))

    except Exception as e:
        raise e

    try:
        #VIT model
        print("####### Test the layout model ##########")
        layout_model_path=Path(r"artifacts\model\layout_model\model.pth")
        #lay=LayoutLMDocumentClassifier(layout_model_path)
        #result=lay.predict(Path("I:\\My Drive\\Work_space\\Project\\document_classification\\dataset\\sample_text_ds\\train\\invoice\\80211531.jpg"))
    
    except Exception as e:
        raise e
    