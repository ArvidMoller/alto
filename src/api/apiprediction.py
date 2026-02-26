import predict

def run_predict():
      name = "model5"         #Name of desired model
      high = 1                #Input range high
      low = -1                #Input range low
      predict_path = "../../Frontend/Media/predict_images"
      info_text = ""          #Info about generated pictures (model settings etc.)
      time_delta = 15         #Time delta between pictures: (has to be a multiple of 15, and match the delta the model was trained on)
      frontend_use = True

      predict.check_predict_img(predict_path, 10, time_delta)

      dataset = predict.load_dataset(predict_path, low, high)

      model = predict.load_model("../models", "model5")

      predicted_sequence = predict.predict_frames(10, model, dataset)

      predict.save_predicted_sequence(predicted_sequence, "predicted_images", name, low, high, info_text, frontend_use)

      predict.plot_predicted_images(dataset, predicted_sequence)