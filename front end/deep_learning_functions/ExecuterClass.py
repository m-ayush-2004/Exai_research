from deep_learning_functions.CompiledClass import *
import keras
def predict(file , disease_name , model_name):
    # Preprocess the uploaded image
    image_array = preprocess_image(file)
    
    # Load the selected model
    model = load_model_from_weights(model_name,disease_name)
    print(image_array.shape)
    # Make prediction using the loaded model
    input_layer = keras.Input(shape=image_array.shape[1:])
    output = model(input_layer)
    model = keras.Model(inputs=input_layer, outputs=output)
    prediction_result = model.predict(image_array)
    print(prediction_result)
    # Generate LIME explanation for the prediction
    segment_info,path,shap_path, lime_path = generate_lime_heatmap_and_explanation(model=model, image=image_array[0],num_segments_to_select=2, model_name=model_name, disease_name=disease_name)
    # Create a Plotly figure with segments and hover information
    # fig = go.Figure()
    # for segment in segment_info:
    #     for contour in segment['contours']:
    #         fig.add_trace(go.Scatter(
    #             x=contour[:, 1],  # X-coordinates
    #             y=contour[:, 0],  # Y-coordinates (inverted)
    #             mode='lines',
    #             line=dict(color=segment['color'], width=2),
    #             hoverinfo='text',
    #             text=f'Segment ID: {segment["id"]}<br>Weight: {segment["weight"]:.4f}',
    #             showlegend=False
    #         ))
    # fig.update_layout(title='LIME Segmentation Visualization', xaxis_title='X', yaxis_title='Y')
    # fig = generate_clinical_visualization(segment_info, image_array)
    # graph_json = fig
    images= fetch_existing_plot_filepaths(model_name,disease_name)
    print(images)
    return images,shap_path, lime_path