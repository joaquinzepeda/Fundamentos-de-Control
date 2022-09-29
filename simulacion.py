# coding=utf-8
"""Simulación de un pendulo invertido
   Code Originally Written by Jacob Hackett, 2022
   Adapted for Instruction by Christian Hubicki

   Adaptado para el curso de Fundamentos de Control de Sistemas
"""

# Cart Pole Code:
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle


__author__ = "Joaquín Zepeda"
__license__ = "MIT"



class simulacion():
    """
    Genera una simulación de un pendulo invertido
    """

    def __init__(self, L=1,m1=2,m2=1,b=0,sim_time=10):
        """
            :param: g gravity (m/s^2)
            :param: L Length (m)
            :param: m1 Cart mass (kg)
            :param: m2 Pendulum mass (kg)
            :param: b Viscous friction coefficient
        """
        self.g = 9.8
        self.L = L
        self.m1 = m1
        self.m2 = m2
        self.b = b

        self.x0 = 0   # Cart Position
        self.dx0 = 0  # Cart Velocity
        self.theta0 = 9.5*np.pi/10 # Pole Angle
        self.dtheta0 = 0  # Pole Angular Velocity

        self.sim_time = sim_time

        self.error_container = []
        self.u_container = []

        self.x_container = []
        self.dx_container = []
        self.theta_container = []
        self.dtheta_container = []

    def set_initial_conditions(self,x0,dx0,theta0,dtheta0):
        """
        ## Choose the starting position of the Cartpole
        # Assign Intial Conditions:
        :param float: x0 Cart Position
        :param float: dx0 Cart Velocity 
        :param angle: theta0 Pole Angle (rad)
        :param angular velocity: dtheta0 Pole Velocity (rad/s)
        """
        
        self.x0 = x0   # Cart Position
        self.dx0 = dx0  # Cart Velocity
        self.theta0 = theta0 # Pole Angle
        self.dtheta0 = dtheta0   # Pole Angular Velocity



    def run_simulation(self,theta_des = np.pi,kp=0,ki=0,kd=0,animate=True,c_color ="purple",p_color="blueviolet"):
        """
        ################################################
        ## Define your Control Input (Force on the Cart)
        # Control input (force)
        #
        # Reference previous measured values 
        # x_vec[i-1]
        # theta_vec[i-1]
        # dx_vec[i-1]
        # dtheta_vec[i-1]

        :param float kp: ganancia proporcional
        :param float ki: ganancia integral
        :param float kd: ganancia derivativa
        :param bool animate: si es True realiza la animación
        :param str c_color: color del carro
        :param str p_color: color del pendulo

        Colores disponibles en https://matplotlib.org/stable/gallery/color/named_colors.html 
        """
        # Define Pendulum Parameters:
        g = 9.8  # gravity (m/s^2)
        L = self.L  # Length (m)
        m1 = self.m1  # Cart mass (kg)
        m2 = self.m2  # Pendulum mass (kg)
        b = self.b  # Viscous friction coefficient

        
        dt = 0.001 #time step
        sim_time = self.sim_time
        t_vec = np.arange(0, sim_time, dt)

        # Initialize State Vectors:
        vec_size = len(t_vec)  # We use this value alot so lets store it
        x_vec = np.zeros(vec_size)
        dx_vec = np.zeros(vec_size)
        theta_vec = np.zeros(vec_size)
        dtheta_vec = np.zeros(vec_size)


        # Pole End effector Location for Animation:
        x_pole = np.zeros(vec_size)
        y_pole = np.zeros(vec_size)

        ################################################
        ## Choose the starting position of the Cartpole
        # Assign Intial Conditions:
        x_vec[0] = self.x0   # Cart Position
        dx_vec[0] = self.dx0  # Cart Velocity
        theta_vec[0] = self.theta0 # Pole Angle
        dtheta_vec[0] = self.dtheta0         # Pole Angular Velocity
        ################################################


        # Initial Pole End effector Location:
        y_offset = 0  # The Y Location of where the Cart and Pole are connected
        x_pole[0] = x_vec[0] + L * np.sin(theta_vec[0])
        y_pole[0] = y_offset - L * np.cos(theta_vec[0])

        # Euler Simulation: Using Matrix Form (A * x = B)
        # Initialize A and B:
        A = np.array([[m1 + m2, 0], [0, 1 / 3 * m2 * L ** 2]])
        B = np.array([0, 0])

        # Integral del error es inicialmente cero
        error_int = 0

        # Simulation Loop
        for i in range(1, vec_size):

            e = theta_des - theta_vec[i-1]
            de_t =  -dtheta_vec[i-1]
            error_int = error_int + e*dt
            u = kp*e+kd*de_t+ ki*error_int


            self.error_container.append(e)
            self.u_container.append(u)
            
            # Simulate the Cart
            # Only the off diagonal needs to be Updated:
            A[0, 1] = 1 / 2 * m2 * L * np.cos(theta_vec[i-1])
            A[1, 0] = 1 / 2 * m2 * L * np.cos(theta_vec[i-1])
            # b must be updated every iteration:
            B[0] = -1 / 2 * m2 * L * (dtheta_vec[i-1] ** 2) * np.sin(theta_vec[i-1]) - b * dx_vec[i-1] + u/m1
            B[1] = -m2 * g * L / 2 * np.sin(theta_vec[i-1])

            [ddx, ddtheta] = np.linalg.solve(A, B)
            # Use ddx and ddtheta to solve:
            x_vec[i] = x_vec[i - 1] + dx_vec[i - 1] * dt 
            theta_vec[i] = theta_vec[i - 1] + dtheta_vec[i - 1] * dt
            dx_vec[i] = dx_vec[i - 1] + ddx * dt
            dtheta_vec[i] = dtheta_vec[i - 1] + ddtheta * dt
            # Extra States for Animation:
            x_pole[i] = x_vec[i] + L * np.sin(theta_vec[i])
            y_pole[i] = y_offset - L * np.cos(theta_vec[i])
        
        self.x_container = x_vec
        self.dx_container = dx_vec
        self.theta_container = theta_vec
        self.dtheta_container = dtheta_vec

        if animate == True:
            # Setup Figure:
            fig, ax = plt.subplots()
            p, = ax.plot([], [], color=p_color)
            min_lim = -10
            max_lim = 10
            ax.axis('equal')
            ax.set_xlim([min_lim, max_lim])
            ax.set_ylim([min_lim, max_lim])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Cartpole Simulation:')
            title = "simulation"

            # Setup Animation Writer:
            FPS = 20
            sample_rate = int(1 / (dt * FPS))  # Real Time Playback
            dpi = 300
            writerObj = FFMpegWriter(fps=FPS)

            # Initialize Patch: (Cart)
            width = 1  # Width of Cart
            height = width / 2  # Height of Cart
            xy_cart = (x_vec[0] - width / 2, y_offset - height / 2)  # Bottom Left Corner of Cart
            r = Rectangle(xy_cart, width, height, color=c_color)  # Rectangle Patch
            ax.add_patch(r)  # Add Patch to Plot

            # Draw the Ground:
            ground = ax.hlines(-height / 2, min_lim, max_lim, colors='black')
            height_hatch = 0.25
            width_hatch = max_lim - min_lim
            xy_hatch = (min_lim, y_offset - height / 2 - height_hatch)
            ground_hatch = Rectangle(xy_hatch, width_hatch, height_hatch, facecolor='None', linestyle='None', hatch='/')
            ax.add_patch(ground_hatch)

            # Animate:
            with writerObj.saving(fig, title + ".mp4", dpi):
                for i in range(0, vec_size, sample_rate):
                    # Update Pendulum Arm:
                    x_pole_arm = [x_vec[i], x_pole[i]]
                    y_pole_arm = [y_offset, y_pole[i]]
                    p.set_data(x_pole_arm, y_pole_arm)
                    # Update Cart Patch:
                    r.set(xy=(x_vec[i] - width / 2, y_offset - height / 2))
                    # Update Drawing:
                    fig.canvas.draw()
                    # Save Frame:
                    writerObj.grab_frame()

    def getSimulationArrays(self):
        """
        Retorna los vectores de posición y velocidad del carro y del péndulo luego de realizada la simulación.
        """
        return [self.x_container,self.dx_container,self.theta_container,self.dtheta_container,self.error_container,self.u_container]

    def reset(self):
        """
        Resetea los valores de la simulación (Condiciones iniciales y los arreglos de la simulación)
        """
        self.x0 = 0   # Cart Position
        self.dx0 = 0  # Cart Velocity
        self.theta0 = 9.5*np.pi/10 # Pole Angle
        self.dtheta0 = 0  # Pole Angular Velocity

        self.error_container = []
        self.u_container = []

        self.x_container = []
        self.dx_container = []
        self.theta_container = []
        self.dtheta_container = []

    def plot_error(self,title="Error con respecto al tiempo"):
        plt.figure()
        plt.plot(self.error_container)
        plt.title(title)
        plt.grid()
        plt.show()

    def plot_control_force(self,title="U con respecto al tiempo"):
        plt.figure()
        plt.plot(self.u_container)
        plt.title(title)
        plt.grid()
        plt.show()

    def plot_x(self,title="Posición del carro con respecto al tiempo"):
        plt.figure()
        plt.plot(self.x_container)
        plt.title(title)
        plt.grid()
        plt.show()

    def plot_dx(self,title="Velocidad del carro con respecto al tiempo"):
        plt.figure()
        plt.plot(self.dx_container)
        plt.title(title)
        plt.grid()
        plt.show()

    def plot_theta(self,title="Ángulo del péndulo con respecto al tiempo"):
        plt.figure()
        plt.plot(self.theta_container)
        plt.title(title)
        plt.grid()
        plt.show()
    
    def plot_dtheta(self,title="Velocidad angular del péndulo con respecto al tiempo"):
        plt.figure()
        plt.plot(self.dtheta_container)
        plt.title(title)
        plt.grid()
        plt.show()
