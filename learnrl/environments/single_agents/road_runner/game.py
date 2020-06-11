from gym import Env, spaces
import numpy as np

class RoadRunnerEnv(Env):

    def __init__(self, n_roads=3, base_speed=0.1, car_prob=None, max_steps=None, car_lenght=0.05):
        """ RoadRunner Gym Environement 
        
        ACTION : {0:nothing, 1:left, 2:right}
        
        """
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=1, shape=(n_roads,)), spaces.Discrete(n_roads)))
        
        self.n_roads = n_roads
        self.base_speed = base_speed
        self.cars = np.array([[self.n_roads // 2, 1.0, self.base_speed]])
        self.position = n_roads // 2

        self.max_steps = int(10/base_speed) if max_steps is None else max_steps
        self.steps = 1

        self.car_prob = 1/n_roads if car_prob is None else car_prob
        self.car_lenght = car_lenght

    def step(self, action):
        if action == 1:
            self.position = (self.position - 1) % self.n_roads
        elif action == 2:
            self.position = (self.position + 1) % self.n_roads
        
        self.generate_car()
        self.move_cars()
        observation = self.get_observation()
        done = observation[0][self.position] <= 1e-6 + self.car_lenght or self.steps >= self.max_steps

        self.remove_cars()
        self.steps += 1
        return observation, 1, done, {}
    
    def generate_car(self):
        if self.car_prob > 1:
            roads = np.random.choice(np.arange(self.n_roads), size=int(self.car_prob), replace=False)
            new_cars = np.array([[road, 1.0, self.base_speed] for road in roads])
            self.cars = np.concatenate((self.cars, new_cars), axis=0)
        elif np.random.rand() < self.car_prob:
            road = np.random.randint(self.n_roads)
            new_cars = np.array([[road, 1.0, self.base_speed]])
            self.cars = np.concatenate((self.cars, new_cars), axis=0)
        
    def move_cars(self):
        self.cars[:, 1] -= self.cars[:, 2]

    def remove_cars(self):
        self.cars = np.delete(self.cars, np.where(self.cars[:, 1] <= 1e-6), axis=0)

    def get_closest_car_distances(self):
        distances = np.ones(self.n_roads)
        for road in range(self.n_roads):
            cars_on_road = self.cars[self.cars[:, 0] == road]
            if len(cars_on_road) > 0:
                distances[road] = np.min(cars_on_road[:, 1])
        return distances
    
    def get_observation(self):
        observation = self.get_closest_car_distances()
        return (np.maximum(0, observation), self.position)
    
    def reset(self):
        self.steps = 1
        self.position = self.n_roads // 2
        self.cars = np.array([[self.n_roads // 2, 1.0, self.base_speed]])
        return self.get_observation()

    def render(self):
        pass
