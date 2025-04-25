import random
import math
import time
import os
#analyse
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import numpy as np
import time
import networkx as nx
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import copy
import curses
##########

train_folder = 'train'
DATA_FILES = [os.path.join(train_folder, f) for f in os.listdir(train_folder) if os.path.isfile(os.path.join(train_folder, f))]

# Výpis dostupných souborů
print("Dostupné soubory:")
for i, file in enumerate(DATA_FILES, start=1):
    print(f"{i}: {file}")

def select_file(prompt):
    while True:
        try:
            selected_index = int(input(prompt)) - 1
            if 0 <= selected_index < len(DATA_FILES):
                return DATA_FILES[selected_index]
            else:
                print("Neplatná volba, zkuste to znovu.")
        except ValueError:
            print("Zadejte prosím číslo.")

# Výběr souboru pro trénovací data
DATA_FILE = select_file("Vyberte číslo souboru pro trénovací data: ")
print(f"Vybraný soubor pro trénovací data: {DATA_FILE}")

# Výběr souboru pro testovací data
TEST_DATA = select_file("Vyberte číslo souboru pro testovací data: ")
print(f"Vybraný soubor pro testovací data: {TEST_DATA}")


# Vstupy od uživatele
INPUT_SIZE = int(input("Zadejte INPUT_SIZE (např. 1): "))
NUM_HIDDEN_LAYERS = int(input("Zadejte NUM_HIDDEN_LAYERS (např. 4): "))
HIDDEN_SIZE = int(input("Zadejte HIDDEN_SIZE (např. 16): "))
OUTPUT_SIZE = int(input("Zadejte OUTPUT_SIZE (např. 1): "))
LEARNING_RATE = float(input("Zadejte LEARNING_RATE (např. 0.0001): "))
EPOCHS = int(input("Zadejte EPOCHS (např. 300): "))
LEAKY_RELU_ALPHA = float(input("Zadejte LEAKY_RELU_ALPHA (např. 0.01): "))
PRINT_EVERY = int(input("Zadejte PRINT_EVERY (např. 1): "))
FUNKCE = input("Zadejte funkci (např. 'y = x * cos(x ^ 2)'): ")

# Rozsah trénovacích dat
ROZSAH_TRAIN_DAT = input("Zadejte ROZSAH_TRAIN_DAT (např. '-8 8'): ")

# Rychlost učení
SPEED = int(input("Zadejte SPEED (např. 70): "))

# Kontrola zadaných hodnot
print("\nNastavení:")
print(f"INPUT_SIZE = {INPUT_SIZE}")
print(f"NUM_HIDDEN_LAYERS = {NUM_HIDDEN_LAYERS}")
print(f"HIDDEN_SIZE = {HIDDEN_SIZE}")
print(f"OUTPUT_SIZE = {OUTPUT_SIZE}")
print(f"LEARNING_RATE = {LEARNING_RATE}")
print(f"EPOCHS = {EPOCHS}")
print(f"LEAKY_RELU_ALPHA = {LEAKY_RELU_ALPHA}")
print(f"PRINT_EVERY = {PRINT_EVERY}")
print(f"FUNKCE = {FUNKCE}")
print(f"ROZSAH_TRAIN_DAT = {ROZSAH_TRAIN_DAT}")
print(f"DATA_FILE = {DATA_FILE}")
print(f"TEST_DATA = {TEST_DATA}")
print(f"SPEED = {SPEED}")


'''''
\\te_x_cos_x216.txt"
\\te_xcos(x)16.txt"
\\tr_(x7+3)2.txt"
\\tr_moc316.txt"
\\tr_x_cos_x216.txt"
\\tr_xcos(x)16.txt"
\\te_(x7+3)2.txt"
\\te_moc316.txt"¨
'''


def he_init_weights(rows, cols):
    """Inicializace vah pomocí He normal (vhodné pro ReLU a Leaky ReLU)"""
    std_dev = math.sqrt(2 / rows)
    return [[random.gauss(0, std_dev) for _ in range(cols)] for _ in range(rows)]

def init_biases(size):
    """Inicializace biasů na nulu (doporučené pro He inicializaci)"""
    return [0.0 for _ in range(size)]

# Inicializace vah a biasů pro vstupní vrstvu do skryté vrstvy
weights = [he_init_weights(INPUT_SIZE, HIDDEN_SIZE)]
biases = [init_biases(HIDDEN_SIZE)]

# Inicializace pro skryté vrstvy
for _ in range(NUM_HIDDEN_LAYERS - 1):
    weights.append(he_init_weights(HIDDEN_SIZE, HIDDEN_SIZE))
    biases.append(init_biases(HIDDEN_SIZE))

# Inicializace pro poslední skrytou vrstvu do výstupní vrstvy
weights.append(he_init_weights(HIDDEN_SIZE, OUTPUT_SIZE))
biases.append(init_biases(OUTPUT_SIZE))



def leaky_relu(x):
    return x if x > 0 else LEAKY_RELU_ALPHA * x

def leaky_relu_derivative(x):
    return 1 if x > 0 else LEAKY_RELU_ALPHA



def forward(x):
    layers = [x]
    for i in range(NUM_HIDDEN_LAYERS + 1):
        next_layer = [leaky_relu(sum(layers[-1][k] * weights[i][k][j] for k in range(len(layers[-1]))) + biases[i][j])
                      for j in range(len(biases[i]))]
        layers.append(next_layer)
    return layers

def backward(x, y, layers):
    global weights, biases
    global errors 

    # Výpočet chyby jako MSE
    errors = [[(layers[-1][i] - y[i]) for i in range(OUTPUT_SIZE)]]
    
    # Gradient MSE: (2/N) * (y_pred - y_true)
    mse_gradient = [(2 / OUTPUT_SIZE) * e for e in errors[0]]
    gradients = [mse_gradient]  # Počáteční gradient chyby

    # Backpropagation skrz vrstvy
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        layer_errors = [sum(gradients[0][j] * weights[i][k][j] for j in range(len(gradients[0]))) 
                        for k in range(len(layers[i]))]
        gradients.insert(0, [layer_errors[k] * leaky_relu_derivative(layers[i][k]) 
                             for k in range(len(layer_errors))])
        
        # Úprava vah a biasů
        for j in range(len(gradients[1])):
            for k in range(len(layers[i])):
                weights[i][k][j] -= LEARNING_RATE * gradients[1][j] * layers[i][k]
            biases[i][j] -= LEARNING_RATE * gradients[1][j]


def load_training_data(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            values = list(map(float, line.split()))
            x = values[:-1]  
            y = values[-1:]  
            data.append((x, y))
    return data



####################### pokus procentualnmi odchylka
def absolute_percentage_error(y_true, y_pred):
    return abs((y_true - y_pred) / y_true) * 100

def calculate_mean_absolute_percentage_error(true_values, predictions):
    errors = [absolute_percentage_error(true, pred) for true, pred in zip(true_values, predictions)]
    return sum(errors) / len(errors)

def mean_squared_error(y_true, y_pred):
    """Vypočítá průměrnou kvadratickou chybu (MSE)"""
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)

def get_weights():
    return copy.deepcopy(weights)  # Hluboká kopie, aby se zachovala historie vah


#######################

#struct
test_data = load_training_data(TEST_DATA)
epoch_weights = []
start_time = time.time()
training_data = load_training_data(DATA_FILE)
gradients_history = []
loss_history = []
epoch_predictions = []

for epoch in range(EPOCHS):
    random.shuffle(training_data)
    total_loss = 0
    for x, y in training_data:
        layers = forward(x)
        loss = mean_squared_error(y, layers[-1])
        total_loss += loss
        backward(x, y, layers)
    
    epoch_weights.append(get_weights())  # Funkce, která vrátí aktuální váhy modelu
    # Zaznamenání predikcí pro testovací data na konci každé epochy
    predictions_epoch = []
    for x, true_y in test_data:
        y_pred = forward(x)[-1]
        predictions_epoch.append(y_pred[0])  # Předpokládám, že y_pred je seznam nebo pole
    epoch_predictions.append(predictions_epoch)

    if (epoch + 1) % PRINT_EVERY == 0:
        avg_loss = total_loss / len(training_data)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.6f}")


end_time = time.time()




#animace
###########################
##################################################################################################
def draw():
    def draw_neural_network(input_size, hidden_layers, hidden_size, output_size):
        G = nx.DiGraph()
        positions = {}
        labels = {}
        layer_sizes = [input_size] + [hidden_size] * hidden_layers + [output_size]
        
        neuron_id = 0
        x_spacing = 2  
        y_spacing = 1.5
        
        for layer_idx, layer_size in enumerate(layer_sizes):
            y_offset = - (layer_size - 1) * y_spacing / 2
            
            for neuron_idx in range(layer_size):
                G.add_node(neuron_id)
                positions[neuron_id] = (layer_idx * x_spacing, y_offset + neuron_idx * y_spacing)
                labels[neuron_id] = f'n{neuron_id+1}'
                neuron_id += 1
        
        prev_layer_start = 0
        for layer_idx in range(len(layer_sizes) - 1):
            current_layer_start = prev_layer_start + layer_sizes[layer_idx]
            
            for i in range(layer_sizes[layer_idx]):
                for j in range(layer_sizes[layer_idx + 1]):
                    G.add_edge(prev_layer_start + i, current_layer_start + j)
            
            prev_layer_start = current_layer_start
        
        plt.figure(figsize=(16, 9), dpi = 160)
        nx.draw(G, pos=positions, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=1500)
        plt.suptitle(f"Struktura neuronové sítě\nNeurony ve skrytých vrstvách: {hidden_size}, Skryté vrstvy: {hidden_layers}")
        plt.subplots_adjust(top=3)
        fig = plt.gcf()
        fig.canvas.manager.full_screen_toggle()
        plt.show(block=True)

    draw_neural_network(INPUT_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_SIZE, OUTPUT_SIZE)
############################################################################################

def animated_pred():
    # Předpokládám, že x_values, true_values a epoch_predictions jsou připraveny
    x_values = np.array([x[0] for x, _ in test_data])
    true_values = np.array([true_y[0] for _, true_y in test_data])

    # Nastavení grafu
    fig, ax = plt.subplots(figsize=(16, 9), dpi = 160)
    line_true, = ax.plot(x_values, true_values, 'bo', label='Skutečné hodnoty')  # Skutečné hodnoty (modré)
    line_pred, = ax.plot([], [], 'ro', label='Predikované hodnoty')  # Predikované hodnoty (červené)

    # Nastavení limitů grafu
    ax.set_xlim(min(x_values), max(x_values))
    ax.set_ylim(min(true_values) - 1, max(true_values) + 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    title = ax.set_title('Predikce během trénování - Epocha: 0')

    # Funkce pro aktualizaci grafu v animaci
    def update(epoch):
        # Predikce pro aktuální epochu
        predictions_epoch = epoch_predictions[epoch]
        
        # Aktualizuj data pro predikce
        line_pred.set_data(x_values, predictions_epoch)

        # Aktualizace nadpisu s aktuální epochou
        title.set_text(f'Predikce během trénování - Epocha: {epoch + 1}/{EPOCHS})')

        return line_pred, title

    # Vytvoření animace
    ani_struct = animation.FuncAnimation(fig, update, frames=EPOCHS, interval=SPEED, blit=False, repeat=False)
    fig.canvas.manager.full_screen_toggle()
    # Zobrazení animace
    plt.show(block=True)


def animated_heatmap():
    # Inicializace grafu (ponecháme colorbar)
    fig, ax = plt.subplots(figsize=(16, 9), dpi = 160)
    norm = Normalize(vmin=-1, vmax=1)
    sm = ScalarMappable(cmap=plt.cm.seismic, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Hodnota váhy (negativní -> pozitivní)')

    # Konstantní parametry
    x_spacing = 2
    y_spacing = 1.5
    layer_sizes = [INPUT_SIZE] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [OUTPUT_SIZE]
    layer_colors = plt.cm.viridis(np.linspace(0, 1, len(layer_sizes)))

    # Předpočítání uzlů a jejich pozic
    positions = {}
    neuron_id = 0
    for layer_idx, layer_size in enumerate(layer_sizes):
        y_offset = -(layer_size - 1) * y_spacing / 2
        for neuron_idx in range(layer_size):
            positions[neuron_id] = (layer_idx * x_spacing, y_offset + neuron_idx * y_spacing)
            neuron_id += 1

    # Předpočítání hran (struktura se nemění)
    edges = []
    edge_segments = []
    prev_layer_start = 0
    for layer_idx in range(len(layer_sizes) - 1):
        current_layer_start = prev_layer_start + layer_sizes[layer_idx]
        for i in range(layer_sizes[layer_idx]):
            for j in range(layer_sizes[layer_idx + 1]):
                edges.append((prev_layer_start + i, current_layer_start + j))
                edge_segments.append([positions[prev_layer_start + i], positions[current_layer_start + j]])
        prev_layer_start = current_layer_start

    # Předpočítání barev uzlů
    node_colors = [layer_colors[layer_idx] for layer_idx in range(len(layer_sizes)) for _ in range(layer_sizes[layer_idx])]

    # Přidání uzlů pomocí scatter
    node_x = [positions[n][0] for n in range(neuron_id)]
    node_y = [positions[n][1] for n in range(neuron_id)]
    scatter_nodes = ax.scatter(node_x, node_y, c=node_colors, s=700)

    # Přidání hran pomocí LineCollection
    edge_collection = LineCollection(edge_segments, cmap=plt.cm.seismic, norm=norm, linewidths=2)
    ax.add_collection(edge_collection)

    def update(epoch):
        """Aktualizuje barvy a tloušťku hran podle vah."""
        weights_list = np.array([
            epoch_weights[epoch][layer_idx][i][j]
            for layer_idx in range(len(layer_sizes) - 1)
            for i in range(layer_sizes[layer_idx])
            for j in range(layer_sizes[layer_idx + 1])
        ])

        edge_collection.set_array(weights_list)  # Aktualizace barev
        edge_collection.set_linewidths(np.abs(weights_list) * 2)  # Aktualizace tloušťky

        ax.set_title(f"Neuronová síť – Vývoj vah (Epocha {epoch+1}/{EPOCHS})")
        print(f"Rendering epoch {epoch+1}/{EPOCHS}")

    fig.canvas.manager.full_screen_toggle()
    ani_heat = animation.FuncAnimation(fig, update, frames=list(range(0, EPOCHS, 1)), repeat=False, interval=SPEED)
    plt.show(block=True)



def animated_loss():
    fig_loss, ax_loss = plt.subplots(figsize=(16, 9), dpi = 160)

    # Inicializace prázdného grafu
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Křivka učení\nneurony ve skrytych vrstvách: {HIDDEN_SIZE}, "
                    f"skryté vrstvy: {NUM_HIDDEN_LAYERS}, alpha: {LEAKY_RELU_ALPHA}, learning_rate: {LEARNING_RATE} "
                    f"\nfunkce: {FUNKCE}, rozsah train dat: {ROZSAH_TRAIN_DAT}")
    ax_loss.grid(True)

    # Počáteční prázdné čáry pro loss křivku
    loss_line, = ax_loss.plot([], [], label="Loss", color="blue")

    # Nastavení os
    ax_loss.set_xlim(0, EPOCHS)  # X osa podle celkového počtu epoch
    ax_loss.set_ylim(0, max(loss_history) * 1.1)  # Y osa podle maxima loss hodnoty

    def update_loss(epoch_idx):
        """Aktualizuje loss graf v závislosti na aktuální epoše."""
        if epoch_idx >= len(loss_history):
            return
        x_data = np.arange(PRINT_EVERY, (epoch_idx + 1) * PRINT_EVERY + 1, PRINT_EVERY)
        y_data = loss_history[:len(x_data)]
        loss_line.set_data(x_data, y_data)

        print(f"Updating loss graph at epoch {epoch_idx+1}")

    ani_loss = animation.FuncAnimation(fig_loss, update_loss, frames=len(loss_history), repeat=False, interval=SPEED)
    fig_loss.canvas.manager.full_screen_toggle()
    plt.legend()
    plt.show(block=True)
    plt.close(fig_loss)  # Zavře okno po skončení animace





print("\ntest:")
predictions = []
true_values = []
x_values1 = []

for x, true_y in test_data:
    y_pred = forward(x)[-1]
    predictions.append(y_pred[0])
    true_values.append(true_y[0])
    x_values1.append(x[0])

    #####################
    # Výpočet průměrné procentuální odchylky
mean_absolute_percentage_error = calculate_mean_absolute_percentage_error(true_values, predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error:.2f}%")
elapsed_time = (end_time - start_time)
print(f"Elapsed time: {elapsed_time:.2f} seconds")
print(type(EPOCHS))
graf_time = round((elapsed_time/EPOCHS), 4)
print(graf_time)
    ################x####

def static():
    x_plot = np.array(x_values1)
    y_true_plot = np.array(true_values)
    y_pred_plot = np.array(predictions)

    plt.figure(figsize=(16, 9), dpi = 160)
    plt.plot(range(PRINT_EVERY, EPOCHS + 1, PRINT_EVERY), loss_history, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Křivka učení\nčas na epochu: {graf_time}, neurony ve skrytych vrstvách: {HIDDEN_SIZE}, skryté vrstvy: {NUM_HIDDEN_LAYERS}, alpha: {LEAKY_RELU_ALPHA}, learning_rate: {LEARNING_RATE} \nfunkce: {FUNKCE}, rozsah train dat: {ROZSAH_TRAIN_DAT}")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, max(loss_history) * 1.1)
    fig = plt.gcf()
    fig.canvas.manager.full_screen_toggle()
    plt.show(block=True)

    plt.figure(figsize=(16, 9), dpi = 160)
    plt.scatter(x_plot, y_true_plot, color='blue', label='skutečné hodnoty')
    plt.scatter(x_plot, y_pred_plot, color='red', label='predikované hodnoty')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f"porovnání skutečné a naučené funkce\nneurony ve skrytych vrstvách: {HIDDEN_SIZE}, skryté vrstvy: {NUM_HIDDEN_LAYERS}, alpha: {LEAKY_RELU_ALPHA}, learning_rate: {LEARNING_RATE} \n funkce: {FUNKCE}, rozsah train dat: {ROZSAH_TRAIN_DAT}")
    plt.grid(True)
    fig = plt.gcf()
    fig.canvas.manager.full_screen_toggle()
    plt.show(block=True)

    print("complete")

def draw_heatmap():
    def draw_network_with_weights(input_size, hidden_layers, hidden_size, output_size, weights):
        G = nx.DiGraph()
        positions = {}
        layer_sizes = [input_size] + [hidden_size] * hidden_layers + [output_size]
        
        neuron_id = 0
        x_spacing = 2
        y_spacing = 1.5
        layer_colors = plt.cm.viridis(np.linspace(0, 1, len(layer_sizes)))

        # Přidání uzlů a jejich pozic
        for layer_idx, layer_size in enumerate(layer_sizes):
            y_offset = -(layer_size - 1) * y_spacing / 2
            for neuron_idx in range(layer_size):
                G.add_node(neuron_id, layer=layer_idx)
                positions[neuron_id] = (layer_idx * x_spacing, y_offset + neuron_idx * y_spacing)
                neuron_id += 1

        # Přidání hran s vahami
        prev_layer_start = 0
        for layer_idx in range(len(layer_sizes) - 1):
            current_layer_start = prev_layer_start + layer_sizes[layer_idx]
            for i in range(layer_sizes[layer_idx]):
                for j in range(layer_sizes[layer_idx + 1]):
                    weight_value = weights[layer_idx][i][j]
                    G.add_edge(prev_layer_start + i, current_layer_start + j, weight=weight_value)
            prev_layer_start = current_layer_start

        # Vizualizace
        edges = G.edges(data=True)
        weights_list = [data['weight'] for _, _, data in edges]
        
        # Normalizace vah pro barvu a tloušťku hran
        norm = Normalize(vmin=-1, vmax=1)

        edge_colors = [plt.cm.seismic(norm(w)) for w in weights_list]  # Červená = negativní, modrá = pozitivní
        edge_widths = [abs(w) * 2 for w in weights_list]  # Zvýraznění síly váhy

        # Barvy uzlů podle vrstev
        node_colors = [layer_colors[G.nodes[n]['layer']] for n in G.nodes()]

        # Vykreslení sítě
        # Vykreslení sítě
        fig, ax = plt.subplots(figsize=(16, 9), dpi = 160)
        nx.draw(
            G, pos=positions, 
            node_color=node_colors, 
            edge_color=edge_colors, 
            width=edge_widths, 
            with_labels=False, 
            node_size=700, 
            edge_cmap=plt.cm.seismic,
            ax=ax  # Explicitně připojíme osy
        )

        # Přidání barevné legendy pro váhy
        sm = ScalarMappable(cmap=plt.cm.seismic, norm=norm)
        sm.set_array([])  # Potřebné pro vytvoření colorbar
        cbar = plt.colorbar(sm, ax=ax)  # Připojení ke konkrétním osám
        cbar.set_label('Hodnota váhy (negativní -> pozitivní)')
        fig = plt.gcf()
        fig.canvas.manager.full_screen_toggle()
        plt.title("Neuronová síť s vizualizací vah")
        plt.axis('off')
        plt.show(block=True)


    # Zavolání funkce
    draw_network_with_weights(INPUT_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_SIZE, OUTPUT_SIZE, weights)

options = ["draw", "draw_heatmap", "animated_heatmap", "static", "animated_loss", "animated_pred"]

def menu(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(0)
    stdscr.timeout(100)
    selected = 0

    while True:
        stdscr.clear()
        stdscr.addstr("Use arrow keys to navigate, Enter to select\n")
        for i, option in enumerate(options):
            if i == selected:
                stdscr.addstr(f"> {option}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f"  {option}\n")

        key = stdscr.getch()
        if key == curses.KEY_UP:
            selected = (selected - 1) % len(options)
        elif key == curses.KEY_DOWN:
            selected = (selected + 1) % len(options)
        elif key == 10:  # Enter key
            stdscr.clear()
            stdscr.addstr(f"Selected: {options[selected]}\n")
            stdscr.refresh()
            curses.napms(1000)
            stdscr.clear()
            globals()[options[selected]]()
            stdscr.addstr("Press any key to return to menu...")
            stdscr.getch()

if __name__ == "__main__":
    curses.wrapper(menu)


