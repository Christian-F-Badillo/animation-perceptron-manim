from manim.utils.color.XKCD import CYAN, LIGHTAQUA
import matplotlib.pyplot as plt
import numpy as np
from manim import *
from sklearn.datasets import make_moons
from utils.utils import (
    initialize_parameters,
    forward_propagation,
    backward_propagation_solution,
    update_parameters,
)
from utils.functionals import compute_cost


class VisualizeNeuralNetworkCategorization(Scene):
    def construct(self) -> None:
        # Parametros y objetos globales
        self.HEADER_FONT_SIZE = 14
        self.HEADER_HEIGHT = -3.5
        self.ANIMATION_RUN_TIME = 0.5

        X_raw, y_raw = make_moons(n_samples=200, noise=0.15, random_state=42)
        X = X_raw.T
        Y = y_raw.reshape(1, -1)

        epochs = 1001
        lr = 0.1
        epoch = 0
        update_animation_interval = 50
        n_x = X.shape[0]
        n_y = Y.shape[0]
        weights = initialize_parameters(n_x, 10, 10, n_y)

        # Create the nodes for each of the layers
        num_nodes = 10
        h1_node_group, h1_nodes_list = self.create_nodes(1.5, 0.25, num_nodes)
        h2_node_group, h2_nodes_list = self.create_nodes(-1.5, 0.25, num_nodes)
        o_node_group, o_nodes_list = self.create_nodes(-4.5, 0.25, 1)

        connections_1 = self.create_connections(
            h1_nodes_list, h2_nodes_list, weights["W2"]
        )
        connections_2 = self.create_connections(
            h2_nodes_list, o_nodes_list, weights["W3"]
        )

        # Animaciones
        axes, labels, data = self.get_data_to_plot(X_raw, y_raw)
        boundaries = self.get_decision_boundary(axes, weights)

        graphing_stuff = VGroup(boundaries, axes, labels, *data)
        self.add(graphing_stuff)

        self.play(
            Create(boundaries),  # Transición en opacidad (Alfa) para la imagen
            Create(axes),  # Trazado de contornos para los ejes
            Write(labels),  # Renderizado de texto
            *(Create(obj) for obj in data),  # Trazado de los puntos
        )
        self.wait(2)
        self.play(graphing_stuff.animate.scale(0.35).shift(LEFT * 5), run_time=2)
        self.wait(2)

        self.play(
            Create(h1_node_group),
            Create(h2_node_group),
            Create(o_node_group),
            Create(connections_1),
            Create(connections_2),
        )
        self.wait(2)

        hidden_layer1_text = self.create_text(
            "Capa Oculta 1", self.HEADER_FONT_SIZE, 1.5, self.HEADER_HEIGHT
        )
        hidden_layer2_text = self.create_text(
            "Capa Oculta 2", self.HEADER_FONT_SIZE, -1.5, self.HEADER_HEIGHT
        )
        output_text = self.create_text(
            "Capa de Salida", self.HEADER_FONT_SIZE, -4.5, self.HEADER_HEIGHT
        )
        status_text = self.create_text(
            f"Época: {0}\n\nPrecisión: {0:.3f}%\n\nError: {0:.3f}",
            self.HEADER_FONT_SIZE + 2,
            -6,
            0,
        )

        self.play(
            Write(hidden_layer1_text),
            Write(hidden_layer2_text),
            Write(output_text),
            Write(status_text),
        )
        self.wait(3)

        for i in range(epochs):
            # Forward pass
            A3, cache = forward_propagation(X, weights)

            pred = (A3 > 0.5).astype(int)
            accuracy = np.mean(pred == Y) * 100

            # Costo
            cost = compute_cost(A3, Y)

            # Backward pass (Derivadas)
            grads = backward_propagation_solution(weights, cache, X, Y)

            # Actualización de pesos
            weights = update_parameters(weights, grads, lr)

            h1, h2, o = (
                np.mean(cache[1], axis=1),
                np.mean(cache[5], axis=1),
                np.mean(cache[9], axis=1),
            )

            if i % update_animation_interval == 0:
                self.animate_nodes(h1_node_group, h1, 1.5, 0.25, num_nodes)
                self.animate_connections(
                    h1_nodes_list, h2_nodes_list, connections_1, weights["W2"]
                )
                self.animate_nodes(h2_node_group, h2, -1.5, 0.25, num_nodes)
                self.animate_connections(
                    h2_nodes_list, o_nodes_list, connections_2, weights["W3"]
                )
                self.animate_nodes(o_node_group, o, -4.5, 0.25, 1)
                self.animate_text(
                    status_text,
                    f"Época: {epoch}\n\nPrecisión: {accuracy:.3f}%\n\nError:{cost:.3f}",
                    self.HEADER_FONT_SIZE + 2,
                    -6,
                    0,
                )
                self.animate_decision_boundary(axes, boundaries, weights)

            epoch += 1

        self.wait(5)

    def get_data_to_plot(self, X, y):
        axes = Axes(
            x_range=[-1.5, 2.75, 0.25],
            y_range=[-2, 2, 0.25],
            axis_config={"color": LIGHTAQUA, "include_numbers": True},
            x_axis_config={
                "numbers_to_include": np.arange(-1.5, 2.75, 0.5),
                "numbers_with_elongated_ticks": np.arange(-1.5, 2.75, 0.5),
            },
            y_axis_config={
                "numbers_to_include": np.arange(-2, 2, 0.5),
                "numbers_with_elongated_ticks": np.arange(-2, 2, 0.5),
            },
            tips=False,
        )

        labels = axes.get_axis_labels(Tex("Eje X").scale(0.7), Tex("Eje Y").scale(0.7))

        mask_0 = y == 0
        mask_1 = y == 1

        x0 = X[mask_0, 0]
        x1 = X[mask_1, 0]
        y0 = X[mask_0, 1]
        y1 = X[mask_1, 1]

        dots_0 = VGroup()
        dots_1 = VGroup()

        for x_coord, y_coord in zip(x0, y0):
            dot = Dot(axes.c2p(x_coord, y_coord), color=RED, radius=0.06)
            dots_0.add(dot)

        for x_coord, y_coord in zip(x1, y1):
            dot = Dot(axes.c2p(x_coord, y_coord), color=CYAN, radius=0.06)
            dots_1.add(dot)

        return axes, labels, (dots_0, dots_1)

    def create_nodes(
        self, left_shift, down_shift, num_nodes, layer_output: np.ndarray | None = None
    ):
        # Create VGroup & list to hold created nodes
        node_group = VGroup()
        nodes = []

        # Create list of circles to represent nodes
        for i in range(num_nodes):
            # Set fill opacity to 0
            opacity = 0.0
            text = "0.000"
            # If a layer output has been passed and the max value is not 0
            if layer_output is not None and np.max(layer_output) != 0.0:
                # Set opacity as normalised layer output value
                opacity = layer_output[i] / np.max(np.absolute(layer_output))
                # Set text as layer output
                text = f"{layer_output[i]:.3f}"

            # Create node
            node = Circle(
                radius=0.26,
                stroke_color=WHITE,
                stroke_width=0.7,
                fill_color=GRAY,
                fill_opacity=opacity,
            )

            # Add to nodes list
            nodes += [node]

            fill_text = Text(text, font_size=12)
            # Position fill text in circle
            fill_text.move_to(node)

            # Group fill text and node and add to node_group
            group = VGroup(node, fill_text)
            node_group.add(group)

        # Arrange & position node_group
        node_group.arrange(DOWN, buff=0.2)
        node_group.shift(left_shift * LEFT).shift(down_shift * DOWN)

        return node_group, nodes

    def create_connections(self, left_layer_nodes, right_layer_nodes, w):
        # Create VGroup to hold created connections
        connection_group = VGroup()

        # Iterate through right layer nodes
        for l in range(len(right_layer_nodes)):
            # Iterate through left layer nodes
            for r in range(len(left_layer_nodes)):
                # Calculate opacity from normalised weight matrix values
                opacity = (
                    0.0
                    if np.max(np.absolute(w[l, :])) == 0.0
                    else w[l, r] / np.max(np.absolute(w[l, :]))
                )
                # Set colour
                colour = GREEN if opacity >= 0 else RED

                # Create connection line
                line = Line(
                    start=right_layer_nodes[l].get_edge_center(LEFT),
                    end=left_layer_nodes[r].get_edge_center(RIGHT),
                    color=colour,
                    stroke_opacity=abs(opacity),
                )

                connection_group.add(line)
        return connection_group

    def create_text(self, text, font_size, left_shift, down_shift):
        # Create text
        text = Text(text, font_size=font_size)

        # Position text
        text.shift(left_shift * LEFT)
        text.shift(down_shift * DOWN)

        return text

    def get_decision_boundary(self, axes, weights, h=0.05):

        x_min, x_max = axes.x_range[0], axes.x_range[1]
        y_min, y_max = axes.y_range[0], axes.y_range[1]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        grid_points = np.c_[xx.ravel(), yy.ravel()].T

        A_out, _ = forward_propagation(grid_points, weights)
        Z = A_out.reshape(xx.shape)

        fig, ax_plt = plt.subplots()
        contour = ax_plt.contour(xx, yy, Z, levels=[0.5])

        boundary_group = VGroup()

        for path in contour.get_paths():
            for segment in path.to_polygons():
                if len(segment) < 2:
                    continue

                manim_points = [axes.c2p(x, y) for x, y in segment]

                curve = VMobject()
                curve.set_points_as_corners(manim_points)
                curve.set_color(WHITE)
                curve.set_stroke(color=WHITE, width=1.5, opacity=0.8)
                curve.set_fill(color=WHITE, opacity=0.08)

                boundary_group.add(curve)

        plt.close(fig)

        boundary_group.set_z_index(1)

        return boundary_group

    def animate_nodes(
        self, layer_group, layer_output, left_shift, down_shift, num_neurons
    ):
        new_layer_group, _ = self.create_nodes(
            left_shift, down_shift, num_neurons, layer_output
        )
        self.play(
            Transform(layer_group, new_layer_group), run_time=self.ANIMATION_RUN_TIME
        )

    def animate_connections(
        self, left_layer_centers, right_layer_centers, line_group, w
    ):
        new_line_group = self.create_connections(
            left_layer_centers, right_layer_centers, w
        )
        self.play(
            Transform(line_group, new_line_group), run_time=self.ANIMATION_RUN_TIME
        )

    def animate_text(self, text, new_string, font_size, left_shift, down_shift):
        new_text = self.create_text(new_string, font_size, left_shift, down_shift)
        self.play(Transform(text, new_text), run_time=self.ANIMATION_RUN_TIME)

    def animate_decision_boundary(self, axes, boundary_group, new_weights, h=0.05):
        new_boundary_group = self.get_decision_boundary(axes, new_weights, h)

        self.play(
            Transform(boundary_group, new_boundary_group),
            run_time=self.ANIMATION_RUN_TIME,
        )

        return new_boundary_group
