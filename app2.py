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
    global selected_zones
    nombre_de_zones = 9  # Nombre de zones souhaitées

    # Créer un dictionnaire pour stocker les points de chaque zone
    zones = {f"zone_{i + 1}": [] for i in range(nombre_de_zones)}
    
    # Liste des couleurs pour chaque zone
    couleurs = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    index_zone = 0
    current_color = couleurs[index_zone]

    # Récupérer le fichier d'image depuis le formulaire
    image_filename = request.form.get('image_filename')
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

    if not os.path.exists(image_path):
        return "Image non trouvée.", 404

    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        return "Erreur lors de la lecture de l'image", 400

    # Callback pour capturer les événements de souris
    def draw_shape(event, x, y, flags, param):
        nonlocal index_zone, current_color

        if event == cv2.EVENT_LBUTTONDOWN:
            # Enregistrer les coordonnées du point où le bouton gauche de la souris est cliqué
            nom_zone_actuelle = f"zone_{index_zone + 1}"
            zones[nom_zone_actuelle].append((x, y))

            # Dessiner un petit cercle à l'endroit où la souris a été cliquée
            cv2.circle(img, (x, y), 3, current_color, -1)
            cv2.imshow('Image', img)

    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback('Image', draw_shape)

    while True:
        cv2.imshow('Image', img)
        key = cv2.waitKey(20)

        if key & 0xFF == ord('s'):
            # Passer à la zone suivante
            index_zone = (index_zone + 1) % nombre_de_zones
            current_color = couleurs[index_zone]
        elif key & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    selected_zones = zones  # Stocker les zones sélectionnées

    # Afficher les points enregistrés pour chaque zone
    print("Zones enregistrées :")
    for nom_zone, points in selected_zones.items():
        print(f"{nom_zone}: {points}")

    # Calculer la chaleur par zone
    zone = []
    chaleur = []
    heatmap_thresholded = cv2.imread(generate_heatmap(image_path), cv2.IMREAD_GRAYSCALE)

    for idx, (zone_name, points_polygone) in enumerate(selected_zones.items()):
        mask = np.zeros(heatmap_thresholded.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points_polygone)], 255)
        selected_heatmap = np.where(mask == 255, heatmap_thresholded, np.nan)
        heat_value = np.nanmean(selected_heatmap)
        zone.append(zone_name)
        chaleur.append(heat_value)

    # Stocker les zones et chaleurs dans la variable globale data_chaleur
    data_chaleur = pd.DataFrame({'Zone': zone, 'Chaleur': chaleur})

    # Rediriger vers la page HTML avec l'image et les zones
    output_image_url = url_for('static', filename=f'uploads/{image_filename}')
    return render_template('select_zones.html', image_url=output_image_url, data_chaleur=data_chaleur)

@app.route('/download-heatmap/<filename>')
def download_heatmap(filename):
    return send_from_directory(app.config['HEATMAP_FOLDER'], filename, as_attachment=True)

@app.route('/create_graph', methods=['POST'])
def create_graph():
    global data_chaleur

    # Récupérer les nouvelles valeurs de zones depuis le formulaire
    for index in range(len(data_chaleur)):
        new_zone_name = request.form.get(f'zone_{index}', data_chaleur.at[index, 'Zone'])
        data_chaleur.at[index, 'Zone'] = new_zone_name

   
    # Générer un graphique à partir des données de chaleur
    plt.figure(figsize=(10, 5))
    plt.bar(data_chaleur['Zone'], data_chaleur['Chaleur'], color='skyblue')
    plt.xlabel('Zones')
    plt.ylabel('Chaleur')
    plt.title('Chaleur par Zone')
    
    # Sauvegarder le graphique
    graph_path = os.path.join(app.config['HEATMAP_FOLDER'], 'chaleur_par_zone.png')
    plt.savefig(graph_path)
    plt.close()

    # Rediriger vers la page avec le graphique
    return render_template('graph.html', graph_image=os.path.basename(graph_path), data=data_chaleur.to_html(classes='data'))

@app.route('/view-graph')
def view_graph():
    return render_template('graph.html')

if __name__ == '__main__':
    # Créer les dossiers si ils n'existent pas
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['HEATMAP_FOLDER'], exist_ok=True)

    # Démarrer l'application Flask
    app.run(debug=True)
