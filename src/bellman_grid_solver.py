"""
Solve deterministic grid worlds/mazes using bellman equation
"""

import time
from collections import defaultdict
from typing import Any, Callable, List, Literal, Optional, Set, Tuple, TypeAlias

import numpy
import pygame
from mazelib import Maze
from mazelib.generate.Prims import Prims
from mazelib.generate.MazeGenAlgo import MazeGenAlgo


Action: TypeAlias = Literal[0, 1, 2, 3]
ActionQValues: TypeAlias = List["QValue"]
Coordinate: TypeAlias = Tuple[int, int]
EnvironmentGenerator: TypeAlias = Callable[[], "GridWorldEnvironment"]
Grid: TypeAlias = numpy.array
QTable: TypeAlias = defaultdict[Coordinate, ActionQValues]
QValue: TypeAlias = float
State: TypeAlias = Coordinate
VisitedSet: TypeAlias = Set[State]
Visualizer: TypeAlias = Callable[[Grid, QTable, Optional[VisitedSet], Callable], None]


class GridWorldEnvironment:
    """
    Represents a grid world environment
    """

    def __init__(
        self,
        grid: Grid,
        start_position: Coordinate,
        goal_position: Coordinate,
    ) -> None:
        """Initialize a GridWorldEnvironment instance

        :param grid: A two dimensional NumPy array filled with -1s and 0s
        :type grid: class:`bellman_grid_solver.Grid`
        :param start_position: A start position for the agent/player
        :type start_position: class:`bellman_grid_solver.Coordinate`
        :param goal_position: A position for the goal that agent/player will reach at the end
        :type goal_position: class:`bellman_grid_solver.Coordinate`
        """

        if len(grid.shape) != 2:
            raise Exception("the given grid should be 2 dimensional")

        grid[goal_position] = 1
        grid[start_position] = 0
        self.grid = grid
        self.start_position = start_position
        self.goal_position = goal_position
        self.state = start_position

    def copy(self) -> "GridWorldEnvironment":
        """Copy the environment states and create a new environment out of them

        :return: A copy of the environment
        :rtype: class:`bellman_grid_solver.GridWorldEnvironment`
        """

        environment_copy = GridWorldEnvironment(
            self.grid, self.start_position, self.goal_position
        )
        environment_copy.state = self.state
        return environment_copy

    def get_next_state(self, action: Action) -> Coordinate:
        """Get the next state based-on action

        :param action: An action
        :type action: class:`bellman_grid_solver.Action`
        :return: The next state after taking the given action
        :rtype: class:`bellman_grid_solver.Coordinate`
        """

        y, x = self.state
        h, w = self.grid.shape

        y += -1 * (action % 2 == 0) + (action == 2) * 2
        x += -1 * (action % 2 != 0) + (action == 3) * 2

        return (min(max(y, 0), h - 1), min(max(x, 0), w - 1))

    def is_terminal_state(self, state: Coordinate) -> bool:
        """Check if state is a termimal state

        :param state: A state
        :type state: class:`bellman_grid_solver.Coordinate`
        :return: True if state is a terminal else False
        :rtype: bool
        """

        return self.grid[state] != 0

    def step(self, action: Action) -> tuple[Coordinate, int, bool]:
        """Do an action

        :param action: An action to be did
        :type action: class:`bellman_grid_solver.Action`
        :return: A tuple containing (next state, reward, is done)
        :rtype: tuple[:class:`bellman_grid_solver.Coordinate`, int, bool]
        """

        next_state = self.get_next_state(action)
        reward = self.grid[next_state]
        is_done = self.is_terminal_state(next_state)
        self.state = next_state
        return next_state, reward, is_done


def generate_maze_environment(
    size: Coordinate, algorithm: type[MazeGenAlgo]
) -> GridWorldEnvironment:
    """Generate a grid world maze environment

    :param size: Height and width of the maze
    :type size: class:`bellman_grid_solver.Coordinate`
    :param algorithm: A maze generation algorithm provided by mazelib
    :type algorithm: class:`mazelib.generate.MazeGenAlgo.MazeGenAlgo`
    :return: A grid world maze environment
    :rtype: class:`bellman_grid_solver.GridWorldEnvironment`
    """

    maze = Maze()
    maze.generator = algorithm(*size)
    maze.generate()
    maze.generate_entrances()
    return GridWorldEnvironment(-maze.grid, maze.start, maze.end)


def get_max_q_value(action_values: ActionQValues) -> float:
    """Get the max Q-value by action values

    :param action_values: A list of Q-values
    :type action_values: class:`bellman_grid_solver.ActionQValues`
    :return: Max Q-value
    :rtype: float
    """

    return max(*action_values)


class GridWorldAgent:
    """
    Represents a grid world agent
    """

    def __init__(self, gamma: float = 0.9) -> None:
        """Initialize a GridWorldAgent instance

        :param gamma: Discount factor, defaults to 0.9
        :type gamma: float, optional
        """

        self.gamma = gamma
        self.q_table: QTable = defaultdict(lambda: [0, 0, 0, 0])
        self._should_stop_computing = False
        self.visualizer: Optional[Visualizer] = None

    def _compute_optimal_policy(
        self, environment: GridWorldEnvironment, visited: Optional[VisitedSet]
    ) -> ActionQValues:
        """Compute the optimal policy

        :param environment: The environment to compute its optimal policy
        :type environment: class:`bellman_grid_solver.GridWorldEnvironment`
        :param visited: A set of visited states for preventing cycles
        :type visited: Optional[:class:`bellman_grid_solver.VisitedSet`]
        :return: A list of four Q-values for each action
        :rtype: class:`bellman_grid_solver.ActionQValues`
        """

        if visited is None:
            visited = set()

        elif environment.state in visited:
            return self._handle_cycle(environment.state)

        visited.add(environment.state)

        if self.visualizer is not None:
            grid = environment.grid.copy()
            grid[environment.state] = 2
            self.visualizer(grid, self.q_table, visited, SimpleEventBus().emit)

        action_results = []
        for action in 0, 1, 2, 3:
            environment_copy = environment.copy()
            step_result = environment_copy.step(action)
            _, reward, is_terminal = step_result

            if self._should_stop_computing:
                action_results.append(0)

            elif not is_terminal:
                future_values = self._compute_optimal_policy(environment_copy, visited)
                discounted_future_value = self.gamma * get_max_q_value(future_values)
                total_value = reward + discounted_future_value
                action_results.append(total_value)

            else:
                self._should_stop_computing = reward >= 1
                action_results.append(reward)

            del environment_copy

        self.q_table[environment.state] = action_results
        visited.remove(environment.state)

        return action_results

    def _handle_cycle(self, state: State) -> ActionQValues:
        """Handle the cycle to prevent exceeding recursion limit

        :param state: A state to get its action Q values
        :type state: class:`bellman_grid_solver.State`
        :return: A list of four Q-values
        :rtype: class:`bellman_grid_solver.ActionQValues`
        """

        return self.q_table[state]

    def get_best_action(self, state: State) -> Action:
        """Get the best action based on what the agent have learned to do on the given state

        :param state: A state to get its argmax Q value
        :type state: class:`bellman_grid_solver.State`
        :return: Best action to take based on optimal policy
        :rtype: class:`bellman_grid_solver.Action`
        """

        return numpy.argmax(self.q_table[state])

    def set_environment(self, environment: GridWorldEnvironment) -> None:
        """Set/Update the environment

        :param environment: An environment to compute its optimal policy
        :type environment: class:`bellman_grid_solver.GridWorldEnvironment`
        """

        self.q_table.clear()
        self._should_stop_computing = False
        self._compute_optimal_policy(environment, None)


def cli_visualize_grid_world(
    grid: Grid,
    q_table: QTable,
    visited: Optional[VisitedSet],
    emit_callbackfn: Callable,
) -> None:
    """Visualize the grid world in CLI

    :param grid: A grid to be visualized
    :type grid: class:`bellman_grid_solver.Grid`
    :param q_table: Current Q-table to be visualized in a way
    :type q_table: class:`bellman_grid_solver.QTable`
    :param visited: A set of visited coordinates or just None
    :type visited: Optional[:class:`bellman_grid_solver.VisitedSet`]
    :param emit_callbackfn: A function for emitting events to be handled by simulations
    :type emit_callbackfn: Callable
    """

    if visited is None:
        visited = set()

    print("\033[2J\033[H", end="")
    for y in range(grid.shape[0]):
        for i in range(2):
            for x in range(grid.shape[1]):
                yx = y, x
                gridyx = grid[y, x]

                if gridyx != 2:
                    if yx in q_table:
                        gridyx = 3

                    elif yx in visited:
                        gridyx = 4

                print(" $@.,#"[gridyx] * 3, end="")
            print()
    time.sleep(0.1)


class PyGameGridWorldVisualizer:
    """
    A PyGame-based grid world visualizer
    """

    def __call__(
        self,
        grid: Grid,
        q_table: QTable,
        visited: Optional[VisitedSet],
        emit_callbackfn: Callable,
    ) -> None:
        """Visualize the grid world in CLI

        :param grid: A grid to be visualized
        :type grid: class:`bellman_grid_solver.Grid`
        :param q_table: Current Q-table to be visualized in a way
        :type q_table: class:`bellman_grid_solver.QTable`
        :param visited: A set of visited coordinates or just None
        :type visited: Optional[:class:`bellman_grid_solver.VisitedSet`]
        :param emit_callbackfn: A function for emitting events to be handled by simulations
        :type emit_callbackfn: Callable
        """

        if visited is None:
            visited = set()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                emit_callbackfn("close")

            elif event.type == pygame.MOUSEBUTTONDOWN:
                emit_callbackfn("toggle_pause")

            elif event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_q:
                        emit_callbackfn("close")

                    case pygame.K_p:
                        emit_callbackfn("toggle_pause")

        self.screen.fill("white")

        if visited:
            self.screen.fill((220, 220, 220))

        screen_size = self.screen.get_size()
        screen_w, screen_h = screen_size

        self.cell_size = (screen_h / grid.shape[0], screen_w / grid.shape[1] / 2)

        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                yx = y, x
                gridyx = grid[yx]

                if gridyx != 2:
                    if yx in q_table:
                        gridyx = 3

                    elif yx in visited:
                        gridyx = 4

                match gridyx:
                    case -1:
                        self.draw_wall(yx)

                    case 1:
                        self.draw_goal(yx)

                    case 2:
                        self.draw_agent(yx)

                    case 3:
                        self.draw_learned(yx)

                    case 4:
                        self.draw_visited(yx)

        self.blit_text(
            (1, grid.shape[1] + 1),
            "Bellman Grid Solver",
            "black",
            False,
            italic=True,
            bold=True
        )
        self.blit_multiline_text(
            (2, grid.shape[1] + 1),
            f"Grid Size: {grid.shape}\n"
            f"Visited States: {len(visited)}\n"
            f"State: {numpy.argwhere(grid == 2)[0]}\n"
            f"Learned States: {len(q_table)}",
            "black",
            False,
        )
        self.blit_text(
            (7, grid.shape[1] + 1),
            "Bindings and Controls",
            "black",
            False,
            italic=True,
            bold=True
        )
        self.blit_multiline_text(
            (8, grid.shape[1] + 1),
            "Press Q to *Close* the window\n"
            "or click the close button on\n"
            "titlebar\n\n"
            "Press P to *Toggle Pause* the\n"
            "visualization\n",
            "black",
            False,
        )
        self.blit_multiline_text(
            (15, grid.shape[1] + 1),
            "(Bindings and Controls doesn't\n"
            "work while the agent is\n"
            "computing the optimal policy)",
            "black",
            False,
            bold=True
        )

        pygame.display.flip()
        self.clock.tick(15)

    def __init__(
        self,
        window_size: Coordinate = (1024, 512),
        window_caption: str = "Grid World Visualization",
    ) -> None:
        """Initialize a PyGameGridWorldVisualizer instance

        :param window_size: Size of the pygame display window
        :type window_size: class:`bellman_grid_solver.Coordinate`
        :param window_caption: Caption/Title of the window
        :type window_caption: str
        """

        pygame.init()

        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption(window_caption)

    def blit_multiline_text(
        self, yx: Coordinate, text: str, color: Any, center_text: bool = True, **kwargs
    ) -> None:
        """Blit the given *multiline* text on screen

        :param yx: Text's position
        :type yx: class:`bellman_grid_solver.Coordinate`
        :param text: Text to be rendered
        :type text: str
        :param color: Color of the text that will be rendered
        :type color: Any
        :param center_text: Should center text or no, defaults to True
        :type center_text: bool, optional
        """

        for subtext in text.split("\n"):
            self.blit_text(yx, subtext, color, center_text, **kwargs)
            y, x = yx
            yx = y + 1, x

    def blit_text(
        self, yx: Coordinate, text: str, color: Any, center_text: bool = True, **kwargs
    ) -> None:
        """Blit the given text on screen

        :param yx: Text's position
        :type yx: class:`bellman_grid_solver.Coordinate`
        :param text: Text to be rendered
        :type text: str
        :param color: Color of the text that will be rendered
        :type color: Any
        :param center_text: Should center text or no, defaults to True
        :type center_text: bool, optional
        """

        y, x = yx
        ch, cw = self.cell_size

        text_surface = self.render_text(text, color, **kwargs)

        if center_text:
            text_rect = text_surface.get_rect()
            text_rect.center = (x * cw + 0.5 * cw, y * ch + 0.5 * ch)
            self.screen.blit(text_surface, text_rect)

        else:
            self.screen.blit(text_surface, (x * cw + 1, y * ch + 1))

    def draw_agent(self, yx: Coordinate) -> None:
        """Draw the agent

        :param yx: Agent's position
        :type yx: class:`bellman_grid_solver.Coordinate`
        """

        y, x = yx
        ch, cw = self.cell_size

        pygame.draw.ellipse(
            self.screen,
            (0, 0, 255),
            (x * cw + 1, y * ch + 1, cw - 2, ch - 2),
        )

    def draw_goal(self, yx: Coordinate) -> None:
        """Draw the goal

        :param yx: Goal's position
        :type yx: class:`bellman_grid_solver.Coordinate`
        """

        self.blit_text(yx, "$", (0, 200, 0))

    def draw_learned(self, yx: Coordinate) -> None:
        """Draw a learned state

        :param yx: State's position
        :type yx: class:`bellman_grid_solver.Coordinate`
        """

        self.blit_text(yx, "L", (200, 0, 200))

    def draw_visited(self, yx: Coordinate) -> None:
        """Draw a visited state

        :param yx: State's position
        :type yx: class:`bellman_grid_solver.Coordinate`
        """

        self.blit_text(yx, "V", (200, 0, 200))

    def draw_wall(self, yx: Coordinate) -> None:
        """Draw a wall

        :param yx: Wall's position
        :type yx: class:`bellman_grid_solver.Coordinate`
        """

        y, x = yx
        ch, cw = self.cell_size

        pygame.draw.rect(
            self.screen, (0, 0, 0), (x * cw + 1, y * ch + 1, cw - 2, ch - 2)
        )

    def render_text(self, text: str, color: Any, **kwargs) -> pygame.surface.Surface:
        """
        Render text using system font `monospace`

        :param text: Text to be rendered
        :type text: str
        :param color: Color of the text that will be rendered
        :type color: Any
        """

        ch, cw = self.cell_size

        font = pygame.font.SysFont("monospace", int(min(ch, cw)), **kwargs)
        return font.render(text, True, color)


class SimpleEventBus:
    """
    A simple event bus implementation
    """

    def __init__(self) -> None:
        """
        Initialize a SimpleEventBus instance
        """

        self.listeners = defaultdict(lambda: [])

    def add_listener(self, name: str, listener: Callable) -> None:
        """Add an event listener

        :param name: Name of event to listen to
        :type name: str
        :param listener: A function thst listens to the event
        :type listener: Callable
        """

        self.listeners[name].append(listener)

    def emit(self, event: str, *args, **kwargs) -> None:
        """Emit an event. Call all the subscribed functions

        :param event: The event to emit
        :type event: str
        """

        for subscriber in self.listeners[event]:
            subscriber(*args, **kwargs)


class GridWorldSimulator:
    """
    A simulator on agent learning how to solve grid world
    """

    def __init__(
        self,
        agent: GridWorldAgent,
        environment_generator: EnvironmentGenerator,
        visualizer: Visualizer,
    ) -> None:
        """Initialize a GridWorldSimulator instance

        :param agent: An agent
        :type agent: class:`bellman_grid_solver.GridWorldAgent`
        :param environment_generator: An environment generator for generating environments
        :type environment_generator: class:`bellman_grid_solver.EnvironmentGenerator`
        :param visualizer: A grid world visualizer
        :type visualizer: class:`bellman_grid_solver.Visualizer`
        """

        self.agent = agent
        self.environment_generator = environment_generator
        self.eventbus = SimpleEventBus()
        self.paused = False
        self.should_close = False
        self.visualizer = visualizer

        self.agent.visualizer = self.visualizer
        self.eventbus.add_listener("close", self.close)
        self.eventbus.add_listener("pause", self.pause)
        self.eventbus.add_listener("resume", self.resume)
        self.eventbus.add_listener("toggle_pause", self.toggle_pause)

    def close(self) -> None:
        """
        Close/dicontinue the simulation
        """

        self.should_close = True

    def generate_environment(self) -> None:
        """
        Generate an environment
        """

        self.environment = self.environment_generator()

    def pause(self) -> None:
        """
        Pause the simulation
        """

        self.paused = True

    def resume(self) -> None:
        """
        Resume the simulation
        """

        self.paused = False

    def simulate(self) -> None:
        """
        Start the simulation
        """

        should_reset_environment = True

        while not self.should_close:
            if should_reset_environment:
                self.generate_environment()
                self.agent.set_environment(self.environment)
                should_reset_environment = False

            grid = self.environment.grid.copy()
            grid[self.environment.state] = 2

            self.visualizer(grid, self.agent.q_table, None, self.eventbus.emit)

            if self.paused:
                continue

            action = self.agent.get_best_action(self.environment.state)
            _, _, is_terminal = self.environment.step(action)
            if is_terminal:
                should_reset_environment = True

    def toggle_pause(self) -> None:
        """
        Toggle pause/resume
        """

        self.paused = not self.paused


def bellman_grid_solver() -> None:
    """
    Solve deterministic grid worlds/mazes using bellman equation
    """

    agent = GridWorldAgent()
    visualizer = PyGameGridWorldVisualizer()
    simulator = GridWorldSimulator(
        agent, lambda: generate_maze_environment((10, 10), Prims), visualizer
    )
    simulator.simulate()


if __name__ == "__main__":
    bellman_grid_solver()
