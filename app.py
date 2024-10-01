import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from scipy.ndimage import zoom
from scipy.special import logsumexp
from DeepGaze import deepgaze_pytorch
from skimage import exposure
from werkzeug.utils import secure_filename
import pandas as pd

# Configurer Flask
app = Flask(__name__)

# Dossier pour stocker les images téléchargées
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['HEATMAP_FOLDER'] = 'static/heatmaps'

# Extensions d'image autorisées
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

DEVICE = 'cpu'
model = deepgaze_pytorch.DeepGazeIIE(pretrained=True).to(DEVICE)

# Variable globale pour stocker les zones sélectionnées
selected_zones = {}

# Vérifier si le fichier est une image autorisée
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Vérifie que le fichier a bien été uploadé
        if 'file' not in request.files:
            return 'Aucun fichier uploadé', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'Aucun fichier sélectionné', 400
        
        # Sauvegarde le fichier uploadé
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        
        # Générer la heatmap
        heatmap_path = generate_heatmap(filepath)
        
        return render_template('result.html', uploaded_image=file.filename, heatmap_image=os.path.basename(heatmap_path))

    return render_template('upload.html')

def generate_heatmap(image_path):
    # Charger et traiter l'image
    image_1 = cv2.imread(image_path)
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image_1, (1024, 1024))  # Redimensionner l'image

    # Charger le modèle de biais de centrage
    centerbias_template = np.load('centerbias_mit1003.npy')
    centerbias = zoom(centerbias_template, (img.shape[0] / centerbias_template.shape[0], img.shape[1] / centerbias_template.shape[1]), order=0, mode='nearest')
    centerbias -= logsumexp(centerbias)

    # Transformer l'image en tenseur
    image_tensor = torch.tensor([img.transpose(2, 0, 1)]).to(DEVICE)
    centerbias_tensor = torch.tensor([centerbias]).to(DEVICE)

    # Prédiction du modèle
    log_density_prediction = model(image_tensor, centerbias_tensor)

    # Générer la heatmap
    heatmap = np.exp(log_density_prediction.detach().cpu().numpy()[0, 0])
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))

    # Redimensionner la heatmap à la taille de l'image originale
    heatmap_resized_dp2 = cv2.resize(heatmap, (image_1.shape[1], image_1.shape[0]))

    # Appliquer un seuil et ajuster le contraste
    heatmap_thresholded = heatmap_resized_dp2.copy()
    threshold = 0.05
    for i in range(heatmap_resized_dp2.shape[0]):
        for j in range(heatmap_resized_dp2.shape[1]):
            if heatmap_resized_dp2[i, j] < threshold:
                heatmap_thresholded[i, j] = 0.03
            else:
                heatmap_thresholded[i, j] = heatmap_resized_dp2[i, j] * 2.5

    heatmap_eq = exposure.equalize_hist(heatmap_thresholded, nbins=256)

    # Sauvegarder l'image originale avec la carte de chaleur superposée
    plt.figure(figsize=(12, 6))
    plt.imshow(image_1)
    plt.imshow(heatmap_eq, alpha=0.7, cmap='jet')
    plt.axis('off')
    
    # Sauvegarder l'image de la carte de chaleur
    heatmap_path = os.path.join(app.config['HEATMAP_FOLDER'], 'heatmap_' + os.path.basename(image_path))
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return heatmap_path

@app.route('/set-zones', methods=['POST'])
def set_zones():
    global data_chaleur  # Pour utiliser la variable globale dans cette fonction

    # Récupérer le nombre de zones et le fichier d'image depuis le formulaire
    nombre_de_zones = request.form.get('nombre_de_zones')
    image_filename = request.form.get('image_filename')

    # Vérification
    if not nombre_de_zones:
        return "Nombre de zones manquant", 400
    if not image_filename:
        return "Aucune image spécifiée", 400

    try:
        nombre_de_zones = int(nombre_de_zones)
    except ValueError:
        return "Nombre de zones invalide", 400

    # Chemin de l'image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

    if not os.path.exists(image_path):
        return "Image non trouvée.", 404

    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        return "Erreur lors de la lecture de l'image", 400

    # Générer la heatmap (ajoutez votre fonction generate_heatmap ici si nécessaire)
    heatmap_path = generate_heatmap(image_path)
    heatmap_thresholded = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

    # Initialisation de la sélection de zones
    zones = {}
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (192, 192, 192)]
    index_zone = 0

    # Callback pour capturer les événements de souris
    def draw_rectangle(event, x, y, flags, param):
        nonlocal img, zones, index_zone, colors

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(zones.get(f'zone_{index_zone + 1}', [])) == 0:
                zones[f'zone_{index_zone + 1}'] = [(x, y)]
            else:
                zones[f'zone_{index_zone + 1}'].append((x, y))
                cv2.rectangle(img, zones[f'zone_{index_zone + 1}'][0], (x, y), colors[index_zone], 2)
                cv2.imshow('Image', img)
                index_zone = (index_zone + 1) % nombre_de_zones

    # Affichage de l'image et sélection des zones
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Image', draw_rectangle)

    while True:
        cv2.imshow('Image', img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    selected_zones = zones

    # Sauvegarde de l'image
    output_image_filename = f"marked_{image_filename}"
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_image_filename)
    cv2.imwrite(output_image_path, img)

    # Calculer la chaleur par zone
    zone = []
    chaleur = []
    for idx, (zone_name, points_polygone) in enumerate(zones.items()):
        mask = np.zeros(heatmap_thresholded.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points_polygone)], 255)
        selected_heatmap = np.where(mask == 255, heatmap_thresholded, np.nan)
        heat_value = np.nanmean(selected_heatmap)
        zone.append(zone_name)
        chaleur.append(heat_value)

    # Stocker les zones et chaleurs dans la variable globale data_chaleur
    data_chaleur = pd.DataFrame({'Zone': zone, 'Chaleur': chaleur})

    # Rediriger vers la page HTML avec l'image et les zones
    output_image_url = url_for('static', filename=f'uploads/{output_image_filename}')
    return render_template('select_zones.html', image_url=output_image_url, data_chaleur=data_chaleur)

    # return render_template('select_zones.html', nombre_de_zones=nombre_de_zones, image_url=image_url, zones=zones, output_image_url=output_image_url)

@app.route('/download-heatmap/<filename>')
def download_heatmap(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename, as_attachment=True)

# Route pour mettre à jour les chaleurs des zones
# @app.route('/update_zones', methods=['POST'])
# def update_zones():
#     global data_chaleur
#     for index, row in data_chaleur.iterrows():
#         new_zone_name = request.form.get(f'zone_{index}', row['Zone'])
#         data_chaleur.at[index, 'Zone'] = new_zone_name
#     return redirect(url_for('set_zones'))

@app.route('/create_graph', methods=['POST'])
@app.route('/create_graph', methods=['POST'])
def create_graph():
    global data_chaleur

    # Récupérer les nouvelles valeurs de zones depuis le formulaire
    for index in range(len(data_chaleur)):
        new_zone_name = request.form.get(f'zone_{index}', data_chaleur.at[index, 'Zone'])
        data_chaleur.at[index, 'Zone'] = new_zone_name

    # Créer le graphique basé sur les nouvelles valeurs
    chart_filename = create_heatmap_chart(data_chaleur)

    # Chemin du graphique généré
    chart_url = url_for('static', filename=f'uploads/{chart_filename}')

    # Chercher l'image qui commence par 'marked' dans le dossier uploads
    uploads_dir = os.path.join(app.static_folder, 'uploads')
    marked_image_filename = None

    for filename in os.listdir(uploads_dir):
        if filename.startswith('marked'):
            marked_image_filename = filename
            break  # On s'arrête après avoir trouvé le premier fichier correspondant

    # Vérifier si une image a été trouvée
    if marked_image_filename:
        output_image_url = url_for('static', filename=f'uploads/{marked_image_filename}')
    else:
        output_image_url = None  # Ou une image par défaut ou un message d'erreur

    # Re-rendre la page `select_zones.html` avec le graphique généré
    return render_template('select_zones.html', image_url=output_image_url, data_chaleur=data_chaleur, chart_url=chart_url)

def create_heatmap_chart(df):
    # Créer et sauvegarder un graphique avec matplotlib
    df = df.sort_values('Chaleur', ascending=True)  # Tri par chaleur

    fig, ax = plt.subplots()
    ax.barh(df['Zone'], df['Chaleur'], color='skyblue')

    # Affichage des valeurs en % à côté des barres
    for i, v in enumerate(df['Chaleur']):
        ax.text(v + 1, i, f"{round(v, 1)} %", color='black', fontweight='bold')

    # Supprimer le cadre autour du graphique
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.xaxis.set_visible(False)
    plt.xlabel("Chaleur (%)")

    # Enregistrer le graphique dans un fichier
    chart_filename = 'heatmap_chart.png'
    chart_path = os.path.join(app.config['UPLOAD_FOLDER'], chart_filename)
    plt.savefig(chart_path)
    plt.close()
    return chart_filename

# @app.route('/display_graph')
# def display_graph():
#     # Afficher le graphique généré à l'utilisateur
#     chart_url = url_for('static', filename='uploads/heatmap_chart.png')
#     return f'<h1>Graphique de Chaleur par Zone</h1><img src="{chart_url}" alt="Graphique de chaleur">'

@app.route('/save-zones', methods=['POST'])
def save_zones():
    global selected_zones
    
    # Afficher les coordonnées des rectangles pour vérification
    for zone_name, points in selected_zones.items():
        print(f"{zone_name}: {points}")

    # Ici, vous pouvez ajouter la logique pour sauvegarder les zones dans une base de données si nécessaire

    return jsonify({"message": "Zones enregistrées avec succès", "zones": selected_zones}), 200

if __name__ == '__main__':
    app.run(debug=True)
