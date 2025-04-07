import numpy as np
import shapely as sh
import matplotlib.pyplot as plt


from urllib3 import request
from multiprocessing import Process, Pipe, cpu_count
from multiprocessing.connection import PipeConnection
from time import time, sleep
from os import remove, mkdir, listdir
from os.path import dirname, abspath, isdir, isfile, getsize
from pathlib import Path
from psutil import virtual_memory

class city:

    def __init__(self, name:str, geocode:int, coords:list[list[float]]) -> None:
        self.name:str = name
        self.geocode:int = geocode
        self.coords:list[list[float]] = coords
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

def cities_gen(file_name:str, state:state) -> list[city]:
    # X, Y, nome, geocodigo, vertex_index, vertex_part, vertex_part_ring, vertex_part_index, distance, angle
    with open(file_name, "r", encoding="utf8") as file:
        line:list[str] = file.readline().split(',')[0:5]
        while (line[0]):
            name:str = line[2]
            geocode:int = int(line[3])
            coords:list[list[float]] = [[float(line[0]),float(line[1])]]
            line = file.readline().split(',')[0:5]
            while (line[0] and int(line[4]) != 0):
                coords.append([float(line[0]),float(line[1])])
                line = file.readline().split(',')[0:5]
            state.add_city(city(name, geocode, coords))
            
    return state.cities

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
            st.add_city(city(name, geocode, coords))
    
    return country.states

def Random_Points_in_Polygon(polygon:sh.Polygon, number:int) -> list[sh.Point]:
    points:list[sh.Point] = []
    minx, miny, maxx, maxy = polygon.bounds
    while (len(points) < number):
        pnt:sh.Point = sh.Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if polygon.contains(pnt):
            points.append(pnt)
    return points

def Hex_Points_in_Polygon(polygon:sh.Polygon, r:float) -> list[sh.Point]:
    xys:np.ndarray = np.column_stack((2*r*0.01*np.round(np.cos(np.array([0, np.pi/3, np.pi*2/3, np.pi, np.pi*4/3, np.pi*5/3])), 6),
                                      2*r*0.01*np.round(np.sin(np.array([0, np.pi/3, np.pi*2/3, np.pi, np.pi*4/3, np.pi*5/3])), 6)))
    points:list[sh.Point] = [polygon.centroid]
    l:int = 1
    ls:int = 0
    while (len(points) > ls):
        ls = len(points)
        cs:np.ndarray = np.tile(np.array([points[0].x, points[0].y]), (6,1))+l*xys
        if (l>1):
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
        ps:list[sh.Point] = list(map(lambda p: sh.Point(p), cs))
        for p in filter(lambda p: polygon.contains(p), ps):
            points.append(p)
        l+=1
    return points

def Hex_Points_in_Polygon_forced(polygon:sh.Polygon, r:float) -> list[sh.Point]:
    xys:np.ndarray = np.column_stack((2*r*0.01*np.round(np.cos(np.array([0, np.pi/3, np.pi*2/3, np.pi, np.pi*4/3, np.pi*5/3])), 6),
                                      2*r*0.01*np.round(np.sin(np.array([0, np.pi/3, np.pi*2/3, np.pi, np.pi*4/3, np.pi*5/3])), 6)))
    points:list[sh.Point] = [polygon.centroid]
    xmin, ymin, xmax, ymax = polygon.bounds
    sqr:sh.Polygon = sh.Polygon([[xmin,ymin], [xmax,ymin], [xmax, ymax], [xmin, ymax]])

    l:int = 1
    ls:int = 0
    while (len(points) > ls):
        ls = len(points)
        cs:np.ndarray = np.tile(np.array([points[0].x, points[0].y]), (6,1))+l*xys
        if (l>1):
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
        ps:list[sh.Point] = list(map(lambda p: sh.Point(p), cs))
        for p in filter(lambda p: sqr.contains(p), ps):
            points.append(p)
        l+=1
    
    return list(filter(lambda p: polygon.contains(p), points))

def city_based_coords_gen(state:state, city:city, r:float, con:PipeConnection):
    city_shape:sh.Polygon = sh.Polygon(city.coords)
    banned:list[int] = [2105104, 2103703, 2105005, 1507904,
                        1502905, 3304904, 3303302, 3304557,
                        4211900, 4206009, 4205407, 3550704,
                        3555406]
    if (not(city_shape.is_valid)):
        if (city.geocode in banned):
            return
        elif (isfile("Vértices\\[%i]_%s_coords_fixed.dat"%(city.geocode, city.name))):
            with open("[5213103]_Mineiros_coords_fixed.dat", 'r') as f:
                city.coords = [[float(l.split(',')[0]), float(l.split(',')[1])] for l in f.readlines()]
        #print(("Shape de %s[%i] não válida")%(city.name, city.geocode))
    
    #start = time()
    rp:list[sh.Point] = Hex_Points_in_Polygon_forced(city_shape, r)
    """ print("Cidade:", city.name, "- ES")
    print("Área: %.2f km²\nRaio: %.2f km\nQtd. de pontos: %i -- esperado: %i"%(city_shape.area*10000, r, len(rp), city_shape.area*10000//(np.pi*r**2)))
    print("execution duration:", time()-start, "\n") """
    
    if (len(rp) < 0.8*city_shape.area*10000//(np.pi*r**2)):
        #print("%s[%i] poucas coordenadas -- %i\nTestando força bruta..."%(city.name, city.geocode, len(rp)))
        rp = Hex_Points_in_Polygon_forced(city_shape, r)
        if (len(rp) < 0.8*city_shape.area*10000//(np.pi*r**2)):
            print("%s[%i] poucas coordenadas: %i -> esperado: %i\tForça bruta falhou..."%(city.name, city.geocode, len(rp), city_shape.area*10000//(np.pi*r**2)))

    if (not(isdir("%s"%(state.sigla)))):
        mkdir("%s"%(state.sigla))
    with open("%s\\[%i]_%s_coords.dat"%(state.sigla, city.geocode, city.name), 'w') as f:
        for p in rp:
            f.write("%f,%f\n"%(p.x, p.y))
    
    con.send(len(rp))
    
    # Plot the polygon
    plt.plot(*city_shape.exterior.xy, scalex=True, scaley=True)
    xp,yp = city_shape.exterior.xy
    plt.plot(xp,yp)

    # Plot the list of points
    xs = [point.x for point in rp]
    ys = [point.y for point in rp]
    plt.scatter(xs,ys,color="red",linewidths=0.001, )
    plt.title("[%i] %s"%(city.geocode, city.name))

    if (not(isdir("Brasil_plots\\%s"%(state.sigla)))):
        mkdir("%s"%(state.sigla))
    with open("%s\\[%i]_%s_coords.dat"%(state.sigla, city.geocode, city.name), 'w') as f:
        for p in rp:
            f.write("%f,%f\n"%(p.x, p.y))

    plt.savefig("Brasil_plots\\%s\\[%i]_%s.png"%(state.sigla, city.geocode, city.name))

def plot_coords(state:state, city:city) -> None:
    city_shape:sh.Polygon = sh.Polygon(city.coords)
    banned:list[int] = [2105104, 2103703, 2105005, 1507904,
                        1502905, 3304904, 3303302, 3304557,
                        4211900, 4206009, 4205407, 3550704,
                        3555406]
    if (not(city_shape.is_valid)):
        if (city.geocode in banned):
            return
        elif (isfile("Vértices\\[%i]_%s_coords_fixed.dat"%(city.geocode, city.name))):
            with open("Vértices\\[%i]_%s_coords_fixed.dat"%(city.geocode, city.name), 'r') as f:
                city.coords = [[float(l.split(',')[0]), float(l.split(',')[1])] for l in f.readlines()]

    stfolder:str = list(filter(lambda f: f[0:2] == state.sigla, listdir("Brasil[1.121.747]")))[0]

    with open("Brasil[1.121.747]\\%s\\[%i]_%s_coords.dat"%(stfolder, city.geocode, city.name)) as f:
        rp:list[sh.Point] = [sh.Point([float(l.split(',')[0]), float(l.split(',')[1][:-2])]) for l in f.readlines()]
    
    # Plot the polygon
    plt.plot(*city_shape.exterior.xy, scalex=True, scaley=True)
    xp,yp = city_shape.exterior.xy
    plt.plot(xp,yp)

    # Plot the list of points
    xs = [point.x for point in rp]
    ys = [point.y for point in rp]
    plt.scatter(xs,ys,color="red",s=0.5)
    plt.title("[%i] %s - %s"%(city.geocode, city.name, state.sigla))

    #draw a circle
    angles:np.ndarray = np.linspace(0, 2*np.pi, 100)
    xs = np.cos(angles)
    ys = np.sin(angles)
    plt.plot(xs, ys, color = 'green')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.gca().set_aspect('equal')
    plt.show()

    if (not(isdir("Brasil_plots\\%s"%(state.sigla)))):
        mkdir("Brasil_plots\\%s"%(state.sigla))

    plt.savefig("Brasil_plots\\%s\\[%i]_%s.svg"%(state.sigla, city.geocode, city.name))


def main_v0():
    Brasil:list[state] = []
    ES = state("Espírito Santo", "ES")
    Brasil.append(ES)
    cities_gen("vert_mun_ES.dat", ES)
    cidade = sh.Polygon(ES.get_cities([3200102])[0].coords)
    #nt = np.array([*ES.get_cities(["Vitória"])[0].coords])
    plt.plot(*cidade.exterior.xy, scalex=True, scaley=True)
    #plt.show()
    print(cidade.area*10000//1)
    rp = Random_Points_in_Polygon(cidade, cidade.area*10000//(np.pi*(1.5)**2))
    print(len(rp))

    # Plot the polygon
    xp,yp = cidade.exterior.xy
    plt.plot(xp,yp)

    # Plot the list of points
    xs = [point.x for point in rp]
    ys = [point.y for point in rp]
    plt.scatter(xs,ys,color="red")
    plt.show()

def main_v1() -> None:
    Brasil:list[state] = []
    ES = state("Espírito Santo", "ES", 32, [])
    Brasil.append(ES)
    cities_gen("vert_mun_ES.dat", ES)
    cidade = sh.Polygon(ES.get_cities([3200102])[0].coords)
    
    r = 1.5
    start = time()
    rp = Hex_Points_in_Polygon(cidade, r)
    print("Cidade:", ES.get_cities([3200102])[0].name, "- ES")
    print("Área: %.2f km²\nRaio: %.2f km\nQtd. de pontos: %i -- esperado: %i"%(cidade.area*10000, r, len(rp), cidade.area*10000//(np.pi*r**2)))
    print("execution duration:", time()-start)


    # Plot the polygon
    plt.plot(*cidade.exterior.xy, scalex=True, scaley=True)
    xp,yp = cidade.exterior.xy
    plt.plot(xp,yp)

    # Plot the list of points
    xs = [point.x for point in rp]
    ys = [point.y for point in rp]
    plt.scatter(xs,ys,color="red",linewidths=0.001)
    plt.show()

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
    states_gen("vert_mun_brasil.dat", Brasil)
    #print([str(s) for s in Brasil.states])

    total_coords:int = 0
    for st in Brasil.get_states([21]):
        start_time:float = time()

        parent_cons:tuple[PipeConnection]
        child_cons:tuple[PipeConnection]
        parent_cons, child_cons = zip(*map(lambda _: Pipe(False), range(len(st.cities))))

        """ with open("[5213103]_Mineiros_coords_fixed.dat", 'r') as f:
            st.get_cities([5213103])[0].coords = [
                [float(l.split(',')[0]), float(l.split(',')[1])] for l in f.readlines()] """

        r:float = 1.35
        processes:list[Process] = list(map(lambda city, con: Process(target=city_based_coords_gen, args=[st, city, r, con]),
                                           st.get_cities([2105104,2103703,2105005]), child_cons))
        
        """ for cit in st.get_cities([2105104,2103703,2105005]):
            with open("[%i]_%s_coords.dat"%(cit.geocode, cit.name), "w") as f:
                f.writelines("%f,%f\n"%(l[0], l[1]) for l in cit.coords) """

        for process in processes:
            process.start()
            sleep(0.1)
            while ((virtual_memory()[0]-virtual_memory()[3])/(1024**2)<311):
                sleep(1)
            
        for process in processes:
            process.join()
            process.close()

        state_coords = sum([con.recv() for con in parent_cons if con.poll()])
        total_coords += state_coords
        print("\nTotal de coordenadas %s: %i"%(st.sigla, state_coords))
        print("execution time %s: %0.6f\n"%(st.sigla, time()-start_time))
        #cont = input("Aperte enter para continuar...")
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
    print(time()-start)

    for st in Brasil.states:
        start_time:float = time()

        processes:list[Process] = list(map(lambda city: Process(target=plot_coords, args=[st, city]), st.cities))
        
        for process in processes:
            process.start()
            while ((virtual_memory()[0]-virtual_memory()[3])/(1024**2)<311):
                sleep(1)
            
        for process in processes:
            process.join()
            process.close()
        
        print("execution time %s: %0.6f\n"%(st.sigla, time()-start_time))
    print("general execution time: %f"%(time()-start_time))

def main_test() -> None:
    #-38.069870023,-11.6063270519999,1,Acajutiba,Não,"2900306",,7,0,0,7,0.01424130176511,100.27803029323688
    start = time()
    ar:np.ndarray = np.loadtxt("%s\\vert_mun_brasil.dat"%(dirname(abspath(__file__))),delimiter=',',usecols=[0,1],encoding="utf-8",dtype=np.ndarray)
    print(time()-start)
    print(ar.shape, ar[0,:])

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
    print(len(lis), lis[0])

if "__main__" ==  __name__:
    main_test()