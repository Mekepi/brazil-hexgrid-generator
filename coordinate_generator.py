import numpy as np
import shapely as sh

from multiprocessing import Process, Pipe, Pool
from multiprocessing.connection import PipeConnection
from time import time, sleep
from os import mkdir, listdir
from os.path import dirname, abspath, isdir, isfile
from pathlib import Path
from psutil import cpu_count

class city:

    def __init__(self, name:str, geocode:int, coords:np.ndarray) -> None:
        self.name:str = name
        self.geocode:int = geocode
        self.coords:np.ndarray = coords
    def __str__(self) -> str:
        return ("%s - %i - %i"%(self.name, self.geocode, len(self.coords)))

class state:

    def __init__(self, name:str, sigla:str, geocode:int, cities:list[city]) -> None:
        self.name:str = name
        self.sigla:str = sigla
        self.geocode:int = geocode
        self.cities:list[city] = cities

    def add_city(self, city:city) -> None:
        if (city in self.cities):
            print("City already included")
        else:
            (self.cities).append(city)

    def get_cities(self, names_or_geocodes:list[str|int]) -> list[city]:
        return list(filter(lambda c: c.geocode in names_or_geocodes or c.name in names_or_geocodes, self.cities))
    
    def __str__(self) -> str:
        return ("%i - %s - %s - %i"%(self.geocode, self.sigla, self.name, len(self.cities)))

class country:

    def __init__(self, name:str, sigla:str, states:list[state], coords:list[list[float]]) -> None:
        self.name:str = name
        self.sigla:str = sigla
        self.states:list[state] = states
        self.coords:list[list[float]] = coords

    def add_states(self, states:list[state]) -> None:
        for state in states:
            if (state in self.states):
                print("%s already inclused"%(state.name))
            else:
                self.states.append(state)

    def get_states(self, names_or_siglas_or_geocodes:list[str|int]) -> list[state]:
        return list(filter(lambda s: s.geocode in names_or_siglas_or_geocodes or s.sigla in names_or_siglas_or_geocodes or s.name in names_or_siglas_or_geocodes, self.states))

def states_gen(file_path:Path, country:country) -> list[state]:
    #X[0],Y[1],fid,nome[3],geometriaaproximada,geocodigo[5],anodereferencia,vertex_index[7],vertex_part,vertex_part_ring,vertex_part_index,distance,angle
    with open(file_path, "r", encoding="utf8") as file:
        line:list[str] = file.readline().split(',')[0:8]
        while (line[0]):
            st:state = country.get_states([int(line[5][1:3])])[0]
            name:str = line[3]
            geocode:int = int(line[5][1:-1])
            coords:list[list[float]] = [[float(line[0]),float(line[1])]]
            line = file.readline().split(',')[0:8]
            while (line[0] and int(line[7]) != 0):
                coords.append([float(line[0]),float(line[1])])
                line = file.readline().split(',')[0:8]
            st.add_city(city(name, geocode, np.array(coords)))
    
    return country.states

def Hex_Points_in_Polygon(polygon:sh.Polygon, r:float) -> np.ndarray:
    xys:np.ndarray = np.column_stack((2*r*0.01*np.round(np.cos(np.array([0, np.pi/3, np.pi*2/3, np.pi, np.pi*4/3, np.pi*5/3])), 6),
                                      2*r*0.01*np.round(np.sin(np.array([0, np.pi/3, np.pi*2/3, np.pi, np.pi*4/3, np.pi*5/3])), 6)))
    points:list[list[float]] = [[polygon.centroid.xy]]
    l:int = 1 #layer
    ls:int = 0
    while (len(points) > ls):
        ls = len(points)
        cs:np.ndarray = np.tile(np.array([points[0][0], points[0][1]]), (6,1))+l*xys
        dxys:np.ndarray = np.array([
                                cs[1,:]-cs[0,:],
                                cs[2,:]-cs[1,:],
                                cs[3,:]-cs[2,:],
                                cs[4,:]-cs[3,:],
                                cs[5,:]-cs[4,:],
                                cs[0,:]-cs[5,:]
        ])
        mcs:np.ndarray = np.repeat(cs, l-1, 0)+np.repeat(dxys/l, l-1, 0)*np.tile(np.reshape(np.arange(1, l, 1), (l-1,1)), (6,1))
        cs = np.concatenate((cs, mcs), 0)
        points.extend(filter(lambda c: polygon.contains(c), cs))
        l+=1
    return np.array(points, dtype=np.float64, ndmin=2)

def Hex_Points_in_Polygon_forced(polygon:sh.Polygon, r:float) -> np.ndarray:
    xys:np.ndarray = np.column_stack((2*r*0.01*np.round(np.cos(np.array([0, np.pi/3, np.pi*2/3, np.pi, np.pi*4/3, np.pi*5/3])), 6),
                                      2*r*0.01*np.round(np.sin(np.array([0, np.pi/3, np.pi*2/3, np.pi, np.pi*4/3, np.pi*5/3])), 6)))
    points:list[list[float]] = [[polygon.centroid.x, polygon.centroid.y]]
    xmin, ymin, xmax, ymax = polygon.bounds
    sqr:sh.Polygon = sh.Polygon([[xmin,ymin], [xmax,ymin], [xmax, ymax], [xmin, ymax]])

    l:int = 1
    ls:int = 0
    while (len(points) > ls):
        ls = len(points)
        cs:np.ndarray = np.tile(np.array([points[0]]), (6,1))+l*xys
        dxys:np.ndarray = np.array([
                                cs[1,:]-cs[0,:],
                                cs[2,:]-cs[1,:],
                                cs[3,:]-cs[2,:],
                                cs[4,:]-cs[3,:],
                                cs[5,:]-cs[4,:],
                                cs[0,:]-cs[5,:]
        ])
        mcs:np.ndarray = np.repeat(cs, l-1, 0)+np.repeat(dxys/l, l-1, 0)*np.tile(np.reshape(np.arange(1, l, 1), (l-1,1)), (6,1))
        cs = np.concatenate((cs, mcs), 0)
        
        points.extend((c for c in cs if sqr.contains(sh.Point([c]))))
        l+=1
    
    return np.array([p for p in points if polygon.contains(sh.Point([p]))], ndmin= 2)

def plot_problem(st:state, cit:city, xys:np.ndarray) -> None:
    import matplotlib.pyplot as plt
    # Plot the polygon
    plt.plot(cit.coords[:,0], cit.coords[:,1], scalex=True, scaley=True, color="#DB5C1F")
    # Plot the list of points
    if(xys.shape[1]):
        plt.scatter(xys[:, 0], xys[:, 1],color="red",s=0.5)
    plt.title("[%i] %s - %s"%(cit.geocode, cit.name, st.sigla))
    plt.show()

def city_based_coords_gen(args) -> int:
    st:state = args[0]
    cit:city = args[1]
    r:float = args[2]

    if (isfile("Vértices\\[%i]_%s_vertex_fixed.dat"%(cit.geocode, cit.name))):
            cit.coords = np.loadtxt("Vértices\\[%i]_%s_vertex_fixed.dat"%(cit.geocode, cit.name), delimiter=',', ndmin=2)
    cit_shape:sh.Polygon = sh.Polygon(cit.coords)

    """ banned:list[int] = [2103703, 1502905, 4211900, 4206009,
                            4205407, 3550704, 3555406]
    if (cit.geocode in banned):
        return 0 """
    
    solved:list[int] = [2605459]
    if (cit.geocode in solved):
        xys:np.ndarray = np.loadtxt("Vértices\\[%i]_%s_coords_fixed.dat"%(cit.geocode, cit.name), delimiter=',', ndmin=2)
    else:
        xys = Hex_Points_in_Polygon_forced(cit_shape, r)

    """ if (xys.shape[0] < 0.8*cit_shape.area*10000//(np.pi*r**2)):
        xys = Hex_Points_in_Polygon_forced(cit_shape, r)
        if (xys.shape[0] < 0.8*cit_shape.area*10000//(np.pi*r**2)):
            print("%s[%i] poucas coordenadas: %i -> esperado: %i\tForça bruta falhou..."%(city.name, city.geocode, xys.shape[0], cit_shape.area*10000//(np.pi*r**2))) """
    
    if (not(isdir("%s"%(st.sigla)))):
        try: mkdir("%s"%(st.sigla))
        except FileExistsError: None
    np.savetxt("%s\\[%i]_%s_coords.dat"%(st.sigla, cit.geocode, cit.name), xys, "%.13f",',')

    #plot_problem(st, cit, xys)
    return xys.shape[0]

def plot_coords(args) -> int:
    st:state = args[0]
    cit:city = args[1]
    if (isfile("Vértices\\[%i]_%s_vertex_fixed.dat"%(cit.geocode, cit.name))):
        cit.coords = np.loadtxt("Vértices\\[%i]_%s_vertex_fixed.dat"%(cit.geocode, cit.name), delimiter=',')

    stfolder:str = next(f for f in listdir("Brasil[1.123.157]") if f[0:2] == st.sigla)

    xys:np.ndarray = np.loadtxt("Brasil[1.123.157]\\%s\\[%i]_%s_coords.dat"%(stfolder, cit.geocode, cit.name), delimiter=',', ndmin=2)

    import matplotlib.pyplot as plt
    from matplotlib import use
    use("svg")

    # Plot the polygon
    plt.plot(cit.coords[:,0], cit.coords[:,1], scalex=True, scaley=True, color="#DB5C1F")

    # Plot the list of points
    if(xys.shape[1]):
        plt.scatter(xys[:, 0], xys[:, 1],color="red",s=0.5)
    plt.title("[%i] %s - %s"%(cit.geocode, cit.name, st.sigla))

    if (not(isdir("Brasil_plots\\%s"%(st.sigla)))):
        try: mkdir("Brasil_plots\\%s"%(st.sigla))
        except FileExistsError: None
    
    plt.savefig("Brasil_plots\\%s\\[%i]_%s.svg"%(st.sigla, cit.geocode, cit.name), backend="svg")
    return 0

def main_v2() -> None:
    Brasil:country = country("Brasil", "BR", [
    state("Acre", "AC", 12, []),
    state("Alagoas", "AL", 27, []),
    state("Amapá", "AP", 16, []),
    state("Amazonas", "AM", 13, []),
    state("Bahia", "BA", 29, []),
    state("Ceará", "CE", 23, []),
    state("Distrito Federal", "DF", 53, []),
    state("Espírito Santo", "ES", 32, []),
    state("Goiás", "GO", 52, []),
    state("Maranhão", "MA", 21, []),
    state("Mato Grosso", "MT", 51, []),
    state("Mato Grosso do Sul", "MS", 50, []),
    state("Minas Gerais", "MG", 31, []),
    state("Pará", "PA", 15, []),
    state("Paraíba", "PB", 25, []),
    state("Paraná", "PR", 41, []),
    state("Pernambuco", "PE", 26, []),
    state("Piauí", "PI", 22, []),
    state("Rio de Janeiro", "RJ", 33, []),
    state("Rio Grande do Norte", "RN", 24, []),
    state("Rio Grande do Sul", "RS", 43, []),
    state("Rondônia", "RO", 11, []),
    state("Roraima", "RR", 14, []),
    state("Santa Catarina", "SC", 42, []),
    state("São Paulo", "SP", 35, []),
    state("Sergipe", "SE", 28, []),
    state("Tocantins", "TO", 17, [])
    ], [])
    states_gen(Path("vert_mun_brasil.dat"), Brasil)

    total_coords:int = 0
    for st in Brasil.states:
        start_time:float = time()
        
        r:float = 1.35
        with Pool(cpu_count()) as p:
            state_coords:int = sum(p.map(city_based_coords_gen, [[st, cit, r] for cit in st.cities]))

        total_coords += state_coords
        print("\nTotal de coordenadas %s: %i"%(st.sigla, state_coords))
        print("execution time %s: %0.6f\n"%(st.sigla, time()-start_time))
    print("Total de coordenadas %s: %i"%(Brasil.name, total_coords))

def main_plot() -> None:
    Brasil:country = country("Brasil", "BR", [
    state("Acre", "AC", 12, []),
    state("Alagoas", "AL", 27, []),
    state("Amapá", "AP", 16, []),
    state("Amazonas", "AM", 13, []),
    state("Bahia", "BA", 29, []),
    state("Ceará", "CE", 23, []),
    state("Distrito Federal", "DF", 53, []),
    state("Espírito Santo", "ES", 32, []),
    state("Goiás", "GO", 52, []),
    state("Maranhão", "MA", 21, []),
    state("Mato Grosso", "MT", 51, []),
    state("Mato Grosso do Sul", "MS", 50, []),
    state("Minas Gerais", "MG", 31, []),
    state("Pará", "PA", 15, []),
    state("Paraíba", "PB", 25, []),
    state("Paraná", "PR", 41, []),
    state("Pernambuco", "PE", 26, []),
    state("Piauí", "PI", 22, []),
    state("Rio de Janeiro", "RJ", 33, []),
    state("Rio Grande do Norte", "RN", 24, []),
    state("Rio Grande do Sul", "RS", 43, []),
    state("Rondônia", "RO", 11, []),
    state("Roraima", "RR", 14, []),
    state("Santa Catarina", "SC", 42, []),
    state("São Paulo", "SP", 35, []),
    state("Sergipe", "SE", 28, []),
    state("Tocantins", "TO", 17, [])
    ], [])
    start = time()
    states_gen(Path("vert_mun_brasil.dat"), Brasil)
    print("states gen duration: ", time()-start, '\n')

    start_time:float = time()
    for st in Brasil.states:
        start_st:float = time()

        r:float = 1.35
        with Pool(cpu_count()) as p:
            state_coords:int = sum(p.map(plot_coords, [[st, cit, r] for cit in st.cities]))
        
        print("execution time %s: %0.6f\n"%(st.sigla, time()-start_st))
    print("general execution time: %f"%(time()-start_time))

def main_test() -> None:
    #-38.069870023,-11.6063270519999,1,Acajutiba,Não,"2900306",,7,0,0,7,0.01424130176511,100.27803029323688
    start = time()
    ar = np.loadtxt("%s\\vert_mun_brasil.dat"%(dirname(abspath(__file__))),delimiter=',',usecols=[0,1],encoding="utf-8")
    print(time()-start)
    
    start = time()
    #winner
    with open("%s\\vert_mun_brasil.dat"%(dirname(abspath(__file__))), 'r', encoding="utf-8") as f:
        l:list[str] = f.readline().split(',')[0:8]
        lis:list[list[float]] = []
        while(l[0]):
            lis.append([float(l[0]), float(l[1])])
            l = f.readline().split(',')[0:8]
    
    """ with open("%s\\vert_mun_brasil.dat"%(dirname(abspath(__file__))), 'r', encoding="utf-8") as f:
        lis:list[list[float]] = []
        for line in f.readlines():
            l:list[str] = line.split(',')
            lis.append([float(l[0]), float(l[1])])
    with open("%s\\vert_mun_brasil.dat"%(dirname(abspath(__file__))), 'r', encoding="utf-8") as f:
        linhas:list[str] = f.readlines()
    lis:list[list[float]] = list(map(lambda l: [float(l[0]), float(l[1])], map(lambda l: l.split(',')[0:8], linhas))) """

    print(time()-start)
    start = time()
    lisar = np.array(lis)
    print(time()-start)

    start = time()
    xs = lisar[:,0]
    ys = lisar[:,1]
    print("%.16f"%(time()-start), xs[0:2], ys[0:2], "-- array[array[float]]")

    start = time()
    xs2 = [l[0] for l in lis]
    ys2 = [l[1] for l in lis]
    print(time()-start, xs2[0:2], ys2[0:2], "-- list[list[float]]")

    lp = [sh.Point((l[0], l[1])) for l in lis]
    
    start = time()
    xs3 = [p.x for p in lp]
    ys3 = [p.y for p in lp]
    print(time()-start, xs3[0:2], ys3[0:2], "-- list[Point]")

    print(len(lisar), lisar[0])

if "__main__" ==  __name__:
    main_plot()
