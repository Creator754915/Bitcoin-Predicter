from tkinter import *
from tkinter import filedialog, messagebox
from tkinter.ttk import Style, Treeview, Entry
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt  # Importation manquante

import pandas as pd
import joblib
from prophet import Prophet

# Déclaration des variables globales
loaded_model = None
predictions = None  # Variable pour stocker les prédictions


def load_model():
    global loaded_model
    filepath = filedialog.askopenfilename(title="Select Model File",
                                          filetypes=(("Pickle files", "*.pkl"), ("All files", "*.*")))
    if filepath:
        loaded_model = joblib.load(filepath)
        status_label.config(text="Model loaded successfully.")
    else:
        status_label.config(text="No model selected.")


graph_canvas = None  # Déclaration de la variable globale graph_canvas


def predict():
    global loaded_model, predictions, graph_canvas  # Utilisation de la variable globale predictions
    if loaded_model:
        # Créer un dataframe avec les données futures
        future_dataframe = loaded_model.make_future_dataframe(periods=365, freq='D')

        # Exemple de données pour les régresseurs utilisés lors de l'entraînement
        example_data = {
            'Volume': [100000, 120000, 110000]  # Exemple de données de volume
            # Ajoutez d'autres régresseurs si nécessaire avec leurs valeurs correspondantes
        }

        # Répéter les valeurs pour chaque jour prédit
        for regressor, values in example_data.items():
            future_dataframe[regressor] = [values[i % len(values)] for i in range(len(future_dataframe))]

        # Faire les prédictions
        predictions = loaded_model.predict(future_dataframe)

        # Afficher les prédictions
        if display_mode.get() == "Graph":
            plot_predictions(predictions)
        elif display_mode.get() == "Table":
            # Supprimer le widget du graphique s'il existe
            if graph_canvas:
                graph_canvas.get_tk_widget().destroy()
            show_predictions_table(predictions)
        else:
            messagebox.showerror("Error", "Invalid display mode selected.")

        status_label.config(text="Predictions generated.")
    else:
        status_label.config(text="No model loaded.")


def plot_predictions(predictions):
    global graph_canvas
    plt.figure(figsize=(8, 4))  # Réduire la taille du graphique
    plt.plot(predictions['ds'], predictions['yhat'], label='Predicted', color='blue')
    plt.fill_between(predictions['ds'], predictions['yhat_lower'], predictions['yhat_upper'], color='skyblue',
                     alpha=0.4)
    plt.xlabel('Date')
    plt.ylabel('BTC Price')
    plt.title('Bitcoin Price Prediction')
    plt.legend()
    plt.grid(True)

    # Intégrer le graphique dans la fenêtre Tkinter
    if graph_canvas:
        graph_canvas.get_tk_widget().destroy()
    graph_canvas = FigureCanvasTkAgg(plt.gcf(), master=graph_frame)
    graph_canvas.draw()
    graph_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)


def show_predictions_table(predictions):
    global treeview
    # Supprimer les précédentes colonnes et lignes de Treeview
    for col in treeview.get_children():
        treeview.delete(col)

    # Afficher les prédictions dans le Treeview
    for index, row in predictions.iterrows():
        treeview.insert("", index, values=(row['ds'], row['yhat'], row['yhat_lower'], row['yhat_upper']))

    # Redimensionner les colonnes du Treeview pour s'adapter au contenu
    for col in ["#0", "#1", "#2", "#3"]:
        treeview.heading(col, text=col, anchor=W)
        treeview.column(col, anchor=W, width=Treeview().heading(col)["width"])


def find_date():
    global predictions  # Utilisation de la variable globale predictions
    search_date = date_entry.get()
    if search_date:
        try:
            search_date = pd.to_datetime(search_date)
            if search_date in predictions['ds'].values:
                data = predictions[predictions['ds'] == search_date]
                # Afficher les informations sur une nouvelle fenêtre
                show_info_window(data)
            else:
                messagebox.showerror("Error", f"No prediction found for {search_date}.")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid date format: {e}")
    else:
        messagebox.showerror("Error", "Please enter a date.")


def show_info_window(data):
    info_window = Toplevel(root)
    info_window.title("Prediction Info")

    # Créer le Treeview pour afficher les informations
    info_treeview = Treeview(info_window, columns=("Date", "Prediction", "Lower Bound", "Upper Bound"), show="headings")
    info_treeview.heading("Date", text="Date")
    info_treeview.heading("Prediction", text="Prediction")
    info_treeview.heading("Lower Bound", text="Lower Bound")
    info_treeview.heading("Upper Bound", text="Upper Bound")
    info_treeview.pack(pady=10, padx=10, fill=BOTH, expand=True)

    # Afficher les prédictions dans le Treeview
    for index, row in data.iterrows():
        info_treeview.insert("", index, values=(row['ds'], row['yhat'], row['yhat_lower'], row['yhat_upper']))


def find_max_value():
    global predictions
    if predictions is not None:
        max_value_row = predictions.loc[predictions['yhat'].idxmax()]
        messagebox.showinfo("Maximum Predicted Value", f"Date: {max_value_row['ds']}\nValue: {max_value_row['yhat']}")
    else:
        messagebox.showerror("Error", "No predictions available.")


def find_min_value():
    global predictions
    if predictions is not None:
        min_value_row = predictions.loc[predictions['yhat'].idxmin()]
        messagebox.showinfo("Minimum Predicted Value", f"Date: {min_value_row['ds']}\nValue: {min_value_row['yhat']}")
    else:
        messagebox.showerror("Error", "No predictions available.")


# Créer la fenêtre principale
root = Tk()
root.title("Prophet Model Predictor")
root.geometry("1000x600")  # Agrandir la fenêtre
root.resizable(False, False)  # Désactiver le redimensionnement de la fenêtre

# Couleur de fond
bg_color = "#f0f0f0"  # Gris clair

# Frame pour afficher le graphique
graph_frame = Frame(root, bg=bg_color)
graph_frame.pack(side=RIGHT, fill=BOTH, expand=True)

# Frame pour les boutons et radiobuttons
button_frame = Frame(root, bg=bg_color)
button_frame.pack(side=LEFT, fill=BOTH, expand=True)

button_color = "#3498db"  # Blue button color

# Bouton pour trouver la valeur prédite la plus haute
max_value_button = Button(button_frame, text="Max Value", command=find_max_value, relief="groove", bg=button_color,
                          width=15)
max_value_button.grid(row=6, column=0, pady=5)

# Bouton pour trouver la valeur prédite la plus basse
min_value_button = Button(button_frame, text="Min Value", command=find_min_value, relief="groove", bg=button_color,
                          width=15)
min_value_button.grid(row=7, column=0, pady=5)

# Chargement du modèle
load_button = Button(button_frame, text="Load Model", command=load_model, relief="groove", bg=button_color, width=15)
load_button.grid(row=0, column=0, pady=5)

# Prédiction
predict_button = Button(button_frame, text="Predict", command=predict, relief="groove", bg=button_color, width=15)
predict_button.grid(row=1, column=0, pady=5)

# Entrée pour la recherche de date
date_entry = Entry(button_frame)
date_entry.grid(row=2, column=0, pady=10)

# Bouton "Find"
find_button = Button(button_frame, text="Find", command=find_date, relief="groove", bg=button_color, width=15)
find_button.grid(row=3, column=0, pady=5)

# Boutons d'affichage
display_mode = StringVar()
display_mode.set("Graph")
graph_button = Radiobutton(button_frame, text="Graph", variable=display_mode, value="Graph")
graph_button.grid(row=4, column=0, padx=10, pady=10)
table_button = Radiobutton(button_frame, text="Table", variable=display_mode, value="Table")
table_button.grid(row=5, column=0, padx=10, pady=10)

# Créer le Treeview
treeview = Treeview(root, columns=("Date", "Prediction", "Lower Bound", "Upper Bound"), show="headings")
treeview.heading("Date", text="Date")
treeview.heading("Prediction", text="Prediction")
treeview.heading("Lower Bound", text="Lower Bound")
treeview.heading("Upper Bound", text="Upper Bound")
treeview.pack(pady=10, padx=10, fill=BOTH, expand=True)

# État
status_label = Label(root, text="", fg="green")
status_label.pack(pady=5)

# Exécuter la boucle d'événements
root.mainloop()
