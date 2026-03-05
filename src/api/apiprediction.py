from predict import combined_loss, check_predict_img, remove_background, download_predict_img, load_dataset, load_model, save_predicted_sequence, predict_frames 
def run_predict():
      print("hello")
      name = "model7"         #Name of desired model
      high = 1                #Input range high
      low = -1                #Input range low
      predict_path = "/satellite_imagery_download/images/predict_images"
      info_text = ""          #Info about generated pictures (model settings etc.)
      time_delta = 15         #Time delta between pictures: (has to be a multiple of 15, and match the delta the model was trained on)
      frontend_use = True

      check_predict_img(predict_path, 10, time_delta)

      dataset = load_dataset(predict_path, low, high)

      model = load_model("/models", name)

      predicted_sequence = predict_frames(10, model, dataset)

      save_predicted_sequence(predicted_sequence, "Frontend/Media/predict_images",time_delta, name, low, high, info_text, frontend_use)