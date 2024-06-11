import matplotlib.pyplot as plt

class Simulador:

    def __init__(self, ubicacion, vel_inicial, tiempo):
        self.g = 9.81
        self.ubicacion = ubicacion
        self.vel_inicial = vel_inicial
        self.tiempo = tiempo
        self.res = None

    def simulacion(self):
        self.res = {}
        for i in range(self.tiempo):
            ubic = self.ubicacion + self.vel_inicial * i - 0.5 * i**2 * self.g
            vel = self.vel_inicial - self.g * i
            self.res[i] = (ubic, vel)

    def plot_vel(self):
        y = [x[1] for x in self.res.values()]
        plt.plot(self.res.keys(), y)
        plt.show()

    def plot_dist(self):
        y = [x[0] for x in self.res.values()]
        plt.plot(self.res.keys(), y)
        plt.show()


def main():
    nsim = int(input("Cuantas simulaciones vas a hacer?"))
    for s in range(nsim):
        print(f"Numero de simulacion {s}")

        s0 = int(input("Ingresa ubicacion inicial (m)"))
        v0 = int(input("Ingresa la velocidad inicial (m/s)"))
        tiempo = int(input("Ingresa el tiempo (s)"))
        sim = Simulador(s0, v0, tiempo)
        sim.simulacion()
        sim.plot_vel()
        sim.plot_dist()


if __name__ == '__main__':
    main()