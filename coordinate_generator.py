import numpy as np
import shapely as sh

from multiprocessing import Pool, Process
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
        self.citiesgc:list[int] = list(c.geocode for c in cities)

    def add_city(self, city:city) -> None:
        if (city.geocode in self.citiesgc):
            #print("City already included")
            None
        else:
            (self.cities).append(city)
            (self.citiesgc).append(city.geocode)

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

def chunck_process(args) -> list[city]:
    #X[0],Y[1],nome[2],geocodigo[3],vertex_index[4]
    cities:list[city] = []
    lines:list[str] = args[0].decode('utf-8').split('\n')
    if (not(lines[-1])):
        lines.pop(-1)
    i:int = 0
    line:list[str] = lines[i].split(',')
    while (i < len(lines)):
        name:str = line[2]
        geocode:int = int(line[3][1:-1])
        coords:list[list[float]] = [[float(line[0]),float(line[1])]]
        i+=1
        line = lines[i].split(',')
        while (i < len(lines) and int(line[4]) != 0):
            coords.append([float(line[0]),float(line[1])])
            i+=1
            if (i < len(lines)): line = lines[i].split(',')
        cities.append(city(name, geocode, np.array(coords)))
    
    return cities

def states_gen_p(ct:country) -> None:
    file_path:Path = Path("%s\\Vértices\\vert_mun_brasil.bin"%(dirname(abspath(__file__))))
    with open(file_path, "rb") as file:
        lines:bytes = file.read()
    
    with Pool(cpu_count()) as p:
        cities_blocks = p.map(
            chunck_process, 
            [
            [lines[        0: 28955665], ct,  1],
            [lines[ 28955665: 58423096], ct,  2],
            [lines[ 58423096: 89149001], ct,  3],
            [lines[ 89149001:118925057], ct,  4],
            [lines[118925057:147005839], ct,  5],
            [lines[147005839:176006902], ct,  6],
            [lines[176006902:204768193], ct,  7],
            [lines[204768193:234440444], ct,  8],
            [lines[234440444:263299591], ct,  9],
            [lines[263299591:292381024], ct, 10],
            [lines[292381024:321690227], ct, 11],
            [lines[321690227:         ], ct, 12]
            ]
        )
    
    for block in cities_blocks:
        for cit in block:
            ct.get_states([cit.geocode//100000])[0].add_city(cit)

def Hex_Points_in_Polygon_forced(polygon:sh.Polygon, r:float) -> np.ndarray:
    xys:np.ndarray = (np.column_stack((2*r*0.01*np.round(np.cos(np.arange(0, 2*np.pi, np.pi/3)), 6),
                                      2*r*0.01*np.round(np.sin(np.arange(0, 2*np.pi, np.pi/3)), 6))))
    points:list[list[float]] = [[polygon.centroid.x, polygon.centroid.y]]
    xmin, ymin, xmax, ymax = polygon.bounds
    sqr:sh.Polygon = sh.Polygon([[xmin,ymin], [xmax,ymin], [xmax, ymax], [xmin, ymax]])
    hex_pts:np.ndarray = np.tile(np.array([points[0]]), (6,1))

    layer:int = 1
    growth_check:int = 0
    while (len(points) > growth_check):
        growth_check = len(points)
        hex_pts += xys
        deltaxys:np.ndarray = np.array(
            [
            hex_pts[1,:]-hex_pts[0,:],
            hex_pts[2,:]-hex_pts[1,:],
            hex_pts[3,:]-hex_pts[2,:],
            hex_pts[4,:]-hex_pts[3,:],
            hex_pts[5,:]-hex_pts[4,:],
            hex_pts[0,:]-hex_pts[5,:]
            ])
        hex_mid_pts:np.ndarray = np.repeat(hex_pts, layer-1, 0)+np.repeat(deltaxys/layer, layer-1, 0)*np.tile(np.reshape(np.arange(1, layer, 1), (layer-1,1)), (6,1))
        hex_all_pts:np.ndarray = np.concatenate((hex_pts, hex_mid_pts), 0)
        
        points.extend((p for p in hex_all_pts if sqr.contains(sh.Point([p]))))
        layer+=1
    
    return np.array([p for p in points if polygon.contains(sh.Point([p]))], ndmin= 2)

def city_based_coords_gen(args) -> int:
    st:state = args[0]
    cit:city = args[1]
    r:float = args[2]

    if (isfile("%s\\Vértices\\[%i]_%s_vertex_fixed.dat"%(dirname(abspath(__file__)), cit.geocode, cit.name))):
            cit.coords = np.loadtxt("%s\\Vértices\\[%i]_%s_vertex_fixed.dat"%(dirname(abspath(__file__)), cit.geocode, cit.name), delimiter=',', ndmin=2)
    cit_shape:sh.Polygon = sh.Polygon(cit.coords)

    xys = Hex_Points_in_Polygon_forced(cit_shape, r)

    if (xys.shape[0] < 0.8*cit_shape.area*10000//(np.pi*r**2)):
        print("%s[%i] poucas coordenadas: %i -> esperado: %i\tForça bruta falhou..."%(cit.name, cit.geocode, xys.shape[0], cit_shape.area*10000//(np.pi*r**2)))
    
    if (not(isdir("%s"%(st.sigla)))):
        try: mkdir("%s"%(st.sigla))
        except FileExistsError: None
    np.savetxt("%s\\%s\\[%i]_%s_coords.dat"%(dirname(abspath(__file__)), st.sigla, cit.geocode, cit.name), xys, "%.13f",',')

    return xys.shape[0]

def plot_coords(args) -> int:
    st:state = args[0]
    cit:city = args[1]
    if (isfile("%s\\Vértices\\[%i]_%s_vertex_fixed.dat"%(dirname(abspath(__file__)), cit.geocode, cit.name))):
        cit.coords = np.loadtxt("%s\\Vértices\\[%i]_%s_vertex_fixed.dat"%(dirname(abspath(__file__)), cit.geocode, cit.name), delimiter=',')

    br:str = next(f for f in listdir(dirname(abspath(__file__))) if f.startswith("Brasil"))
    stfolder:str = next(f for f in listdir("%s\\%s"%(dirname(abspath(__file__)), br)) if f[0:2] == st.sigla)

    xys:np.ndarray = np.loadtxt("%s\\%s\\%s\\[%i]_%s_coords.dat"%(dirname(abspath(__file__)), br, stfolder, cit.geocode, cit.name), delimiter=',', ndmin=2)

    import matplotlib.pyplot as plt
    from matplotlib import use
    use("agg")

    # Plot the polygon
    plt.plot(cit.coords[:,0], cit.coords[:,1], scalex=True, scaley=True, color="#DB5C1F")

    # Plot the list of points
    if(xys.shape[1]):
        plt.scatter(xys[:, 0], xys[:, 1],color="red",s=0.5)
    plt.title("[%i] %s - %s"%(cit.geocode, cit.name, st.sigla))

    if (not(isdir("%s\\plots"%(dirname(abspath(__file__)))))):
        try: mkdir("%s\\plots"%(dirname(abspath(__file__))))
        except FileExistsError: None
    if (not(isdir("%s\\plots\\%s"%(dirname(abspath(__file__)), st.sigla)))):
        try: mkdir("%s\\plots\\%s"%(dirname(abspath(__file__)), st.sigla))
        except FileExistsError: None
    
    #plt.show()
    plt.savefig("%s\\plots\\%s\\[%i]_%s.png"%(dirname(abspath(__file__)), st.sigla, cit.geocode, cit.name), backend='agg', dpi=200)
    return 0

def brasil_gen() -> country:
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
    #start = time()
    states_gen_p(Brasil)
    #print(time()- start)

    return Brasil

def main_gen(r:float) -> None:
    Brasil:country = brasil_gen()
    total_coords:int = 0
    for st in Brasil.states:
        start_time:float = time()
        
        with Pool(cpu_count()) as p:
            state_coords:int = sum(p.map(city_based_coords_gen, [[st, cit, r] for cit in st.cities]))

        total_coords += state_coords
        print("\nTotal de coordenadas %s: %i"%(st.sigla, state_coords))
        print("execution time %s: %0.6f\n"%(st.sigla, time()-start_time))
    print("Total de coordenadas %s: %i"%(Brasil.name, total_coords))

def main_plot(interval:float=0.1) -> None:
    Brasil:country = brasil_gen()
    start_time:float = time()
    for st in Brasil.states:
        start_st:float = time()

        processes:list[Process] = list(map(lambda cit: Process(target=plot_coords, args=[[st, cit]]), st.cities))
        
        for process in processes:
            process.start()
            sleep(interval)
        
        for process in processes:
            process.join()
            process.close()
        """ with Pool(cpu_count()) as p:
            _:int = sum(p.map(plot_coords, [[st, cit] for cit in st.cities])) """
        
        print("plot time %s: %0.6f\n"%(st.sigla, time()-start_st))
    print("general plot time: %f"%(time()-start_time))

def main() -> None:
    """ Gerador de coordenadas. Recebe apenas o raio em Km.
        Gerará pastas para todo os estados. Depois só tacar de novo na pasta Brasil que tinhas as coordenadas previamente. """
    
    #main_gen(1.35)

    """ Gerador de gráficos pra visualização tanto do formato do município como da qualidade das coordenadas geradas.
        Não recebe nenhum parametro e gera png's de todos os os municípios do país na pasta gerada 'plot'.
        Caso queira outro formato, alterar na função plot_coords.

        Se, por algum acaso, os plots estejam estrenhos, é por conta do intervalo entre cada plot. Matplotlib não lida bem com paralelismo. 
        Então só pôr como argumento da função main_plot um valor maior que o base, que é 0.1 (isso são segundos entre o início de cada processo).
        Como geralmente leva 0.6~0.8s para um plot completo terminar, então, sendo conservador, 0.4 já garante total estabilidade. """
    
    main_plot()

    """ Tanto country, como state tem funções getters. Caso queria acessar apenas alguns estados ou municípios em específico. """

    #country.get_states(["BA", 13]) #Recebe lista de geocódigo de estados ou siglas

    #state.get_cities([1200013, 1300029]) #Recebe lista de geocódigos



if "__main__" ==  __name__:
    main()
