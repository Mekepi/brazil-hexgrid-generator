import numpy as np
import shapely as sh

from multiprocessing import Pool
from time import time, perf_counter
from os import mkdir, listdir, rename
from os.path import dirname, abspath, isdir
from pathlib import Path
from psutil import cpu_count

class city:

    def __init__(self, name:str, geocode:int, coords:np.ndarray) -> None:
        self.name:str = name
        self.geocode:int = geocode
        self.vertices:np.ndarray = coords
    def __str__(self) -> str:
        return ("[%i] %s - vertices: %i"%(self.geocode, self.name, len(self.vertices)))

class state:

    def __init__(self, name:str, sigla:str, geocode:int, cities:list[city]) -> None:
        self.name:str = name
        self.sigla:str = sigla
        self.geocode:int = geocode
        self.cities:list[city] = cities
        self.citiesgc:list[int] = list(c.geocode for c in cities)

    def add_city(self, city:city) -> None:
        if (city.geocode in self.citiesgc):
            print("[%i] %s already included in %s"%(city.geocode, city.name, self.sigla))
        else:
            (self.cities).append(city)
            (self.citiesgc).append(city.geocode)

    def get_cities(self, names_or_geocodes:list[str|int]) -> list[city]:
        return [cit for cit in self.cities if (cit.geocode in names_or_geocodes or cit.name in names_or_geocodes)]
    
    def __str__(self) -> str:
        return ("[%i] %s - %s - cities:%i"%(self.geocode, self.sigla, self.name, len(self.cities)))

class country:

    def __init__(self, name:str, sigla:str, states:list[state], coords:list[list[float]]) -> None:
        self.name:str = name
        self.sigla:str = sigla
        self.states:list[state] = states
        self.coords:list[list[float]] = coords

    def add_states(self, states:list[state]) -> None:
        for state in states:
            if (state in self.states):
                print("%s already inclused in %s"%(state.name, self.name))
            else:
                self.states.append(state)

    def get_states(self, names_or_siglas_or_geocodes:list[str|int]) -> list[state]:
        return [st for st in self.states if (st.geocode in names_or_siglas_or_geocodes or st.sigla in names_or_siglas_or_geocodes or st.name in names_or_siglas_or_geocodes)]
    
    def __str__(self) -> str:
        return("%s - %s - vertices: %i\nstates: %i"%(self.sigla, self.name, len(self.coords), len(self.states)))

def chunck_process(args) -> list[city]:
    #X[0],Y[1],nome[2],geocodigo[3],vertex_index[4]
    cities:list[city] = []
    lines:list[bytes] = args[0].splitlines()
    

    i:int = 0
    llen:int = len(lines)
    line:list[bytes] = lines[i].split(b',')
    while (i < llen):
        name:str = line[2].decode('utf-8')
        geocode:int = int(line[3][1:-1])
        vertices:list[list[float]] = [[float(line[0]),float(line[1])]]
        i+=1
        line = lines[i].split(b',')
        while (i < llen and int(line[4]) != 0):
            vertices.append([float(line[0]),float(line[1])])
            i+=1
            if (i < llen): line = lines[i].split(b',')
        cities.append(city(name, geocode, np.array(vertices)))
    
    return cities

def states_gen_p(ct:country) -> None:
    file_path:Path = Path("%s\\Vértices\\vert_mun_brasil.bin"%(dirname(abspath(__file__))))
    with open(file_path, "rb") as file:
        lines:bytes = file.read()
    
    with Pool(cpu_count()) as p:
        cities_blocks = p.map(
            chunck_process, 
            [
            [lines[        0: 28955665]],
            [lines[ 28955665: 58423096]],
            [lines[ 58423096: 89149001]],
            [lines[ 89149001:118925057]],
            [lines[118925057:147005839]],
            [lines[147005839:176006902]],
            [lines[176006902:204768193]],
            [lines[204768193:234440444]],
            [lines[234440444:263299591]],
            [lines[263299591:292381024]],
            [lines[292381024:321690227]],
            [lines[321690227:         ]]
            ]
        )
    
    for block in cities_blocks:
        for cit in block:
            ct.get_states([cit.geocode//100000])[0].add_city(cit)

    for gc in [fv[1:8] for fv in listdir("%s\\Vértices"%(dirname(abspath(__file__)))) if fv.endswith("vertex_fixed.dat")]:
        c:city = ct.get_states([int(gc[:2])])[0].get_cities([int(gc)])[0] 
        c.vertices = np.loadtxt("%s\\Vértices\\[%i]_%s_vertex_fixed.dat"%(dirname(abspath(__file__)), c.geocode, c.name), delimiter=',')

    with open("p1_list_orig.txt", "rb") as f:
        p1_list = (ct.get_states([int(gc)//100000])[0].get_cities([int(gc)])[0] for gc in f.read().split(b','))
    for cit in p1_list:
        poly:sh.Polygon = sh.simplify(sh.Polygon(cit.vertices), 0.01)
        cit.vertices = np.array(poly.exterior.coords, ndmin=2)

def Hex_Points_in_Polygon_forced(cit:city, r:float) -> np.ndarray:
    polygon:sh.Polygon = sh.Polygon(cit.vertices)
    xys:np.ndarray = (np.column_stack((2*r*0.01*np.round(np.cos(np.arange(0, 2*np.pi, np.pi/3)), 6),
                                      2*r*0.01*np.round(np.sin(np.arange(0, 2*np.pi, np.pi/3)), 6))))
    points:list[list[float]] = [[polygon.centroid.x, polygon.centroid.y]]
    rect:sh.Polygon = sh.minimum_rotated_rectangle(polygon)
    sh.prepare(rect)

    """ with open("p1_list_orig.txt", "rb") as f:
        p1_list:list[int] = [int(gc) for gc in f.read().split(b',')]
    if (cit.geocode in p1_list):
        polygon = sh.simplify(polygon, 0.02) """
    sh.prepare(polygon)

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
        
        points.extend(hex_all_pts[np.vectorize(sh.within, signature='(a),()->(a)')(np.vectorize(sh.Point,signature='(a)->()')(hex_all_pts), rect)])
        layer+=1
    
    pa:np.ndarray = np.array(points)
    return pa[np.vectorize(sh.within, signature='(a),()->(a)')(np.vectorize(sh.Point,signature='(a)->()')(pa), polygon)]

def city_based_coords_gen(args) -> int:
    st:state = args[0]
    cit:city = args[1]
    r:float = args[2]
    
    cit_shape:sh.Polygon = sh.Polygon(cit.vertices)

    start = perf_counter()
    xys = Hex_Points_in_Polygon_forced(cit, r)
    end = perf_counter()
    """ if(end-start>2.):
        with open("p1_list_log.txt", "a", encoding="utf-8") as f:
            f.write("[%i] %s hex_gen time (coords: %i, area: %.2f, prod: %.1f): %2.3f\n"%(cit.geocode, cit.name, len(cit.vertices), cit_shape.area, len(cit.vertices)*cit_shape.area, end-start))
        with open("p1_list.txt", "a", encoding="utf-8") as f:
            f.write("%i,"%(cit.geocode)) """
    """ if (xys.shape[0] < 0.8*cit_shape.area*10000//(np.pi*r**2)):
        #print("%s[%i] poucas coordenadas: %i -> esperado: %i\tForça bruta falhou..."%(cit.name, cit.geocode, xys.shape[0], cit_shape.area*10000//(np.pi*r**2)))
        with open("alc.txt", 'a') as f:
            f.write("%i,"%(cit.geocode))
        with open("acceptable low coords log.txt", 'a', encoding='utf-8') as f:
            f.write("%s[%i] poucas coordenadas: %i -> esperado: %i\tForça bruta falhou...\n"%(cit.name, cit.geocode, xys.shape[0], cit_shape.area*10000//(np.pi*r**2))) """
    
    
    if (not(isdir("%s\\coords"%(dirname(abspath(__file__)))))):
        try: mkdir("%s\\coords"%(dirname(abspath(__file__))))
        except FileExistsError: None
    if (not(isdir("%s\\coords\\%s"%(dirname(abspath(__file__)), st.sigla)))):
        try: mkdir("%s\\coords\\%s"%(dirname(abspath(__file__)), st.sigla))
        except FileExistsError: None
    np.savetxt("%s\\coords\\%s\\[%i]_%s_coords.dat"%(dirname(abspath(__file__)), st.sigla, cit.geocode, cit.name), xys, "%.13f",',')

    return xys.shape[0]

def plot_coords(args) -> None:
    st:state = args[0]
    cit:city = args[1]

    br:str = next(f for f in listdir(dirname(abspath(__file__))) if f.startswith("Brasil"))
    stfolder:str = next(f for f in listdir("%s\\%s"%(dirname(abspath(__file__)), br)) if f[0:2] == st.sigla)

    xys:np.ndarray = np.loadtxt("%s\\%s\\%s\\[%i]_%s_coords.dat"%(dirname(abspath(__file__)), br, stfolder, cit.geocode, cit.name), delimiter=',', ndmin=2)

    import matplotlib.pyplot as plt
    from matplotlib import use
    use("Agg")

    # Plot the polygon
    plt.plot(cit.vertices[:,0], cit.vertices[:,1], scalex=True, scaley=True, color="#DB5C1F")
    
    # Plot the list of points
    plt.scatter(xys[:, 0], xys[:, 1],color="red",s=0.5)
    plt.title("[%i] %s - %s"%(cit.geocode, cit.name, st.sigla))

    if (not(isdir("%s\\plots"%(dirname(abspath(__file__)))))):
        try: mkdir("%s\\plots"%(dirname(abspath(__file__))))
        except FileExistsError: None
    if (not(isdir("%s\\plots\\%s"%(dirname(abspath(__file__)), st.sigla)))):
        try: mkdir("%s\\plots\\%s"%(dirname(abspath(__file__)), st.sigla))
        except FileExistsError: None
    
    #plt.show()
    plt.savefig("%s\\plots\\%s\\[%i]_%s.png"%(dirname(abspath(__file__)), st.sigla, cit.geocode, cit.name), backend='Agg', dpi=200)
    plt.close()

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
    start = time()
    states_gen_p(Brasil)
    print(time()- start)

    return Brasil

def main_gen(r:float) -> None:
    Brasil:country = brasil_gen()
    start:float = time()
    total_coords:int = 0
        
    with Pool(cpu_count()) as p:
        for st in Brasil.states:
            start_time:float = time()
            """ with open("p1_list_orig.txt", "rb") as f:
                p1_list:list[int|str] = [int(gc) for gc in f.read().split(b',')] """
            state_coords:int = sum(p.map(city_based_coords_gen, [[st, cit, r] for cit in st.cities]))

            #rename("%s\\coords\\%s"%(dirname(abspath(__file__)), st.sigla), "%s\\coords\\%s[%i]"%(dirname(abspath(__file__)), st.sigla, state_coords))

            total_coords += state_coords
            print("\nTotal de coordenadas %s: %i"%(st.sigla, state_coords))
            print("execution time %s: %0.6f\n"%(st.sigla, time()-start_time))

    #rename("%s\\coords"%(dirname(abspath(__file__))), "%s\\coords[%i]"%(dirname(abspath(__file__)), total_coords))

    print("Total de coordenadas %s: %i"%(Brasil.name, total_coords))
    print("execution time:", time()-start)

def main_plot() -> None:
    Brasil:country = brasil_gen()
    start_time:float = time()

    with Pool(cpu_count()) as p:
        for st in Brasil.states:
            start_st:float = time()
            with open("p1_list_orig.txt", "rb") as f:
                p1_list:list[int|str] = [int(gc) for gc in f.read().split(b',')]
            p.map(plot_coords, [[st, cit] for cit in st.get_cities(p1_list)])
            #p.map(plot_coords, [[st, cit] for cit in st.cities], (len(st.cities)//cpu_count())+1)
            print("plot time %s: %0.6f\n"%(st.sigla, time()-start_st))

    print("general plot time: %f"%(time()-start_time))

def main() -> None:
    """ Gerador de coordenadas. Recebe apenas o raio em Km.
        Gerará pastas para todo os estados dentro da pasta coords. Depois só o nome da pasta para começar com Brasil.
        Caso queira que as pastas gere o número de coordenadas, basta descomentar os renames na função main_gen. 
        Porém lembrar de comentá-los ou mudar o nome da pasta coords para não gerar erros em próximas execuções. """
    
    #brasil_gen()
    main_gen(1.35)

    """ Gerador de gráficos pra visualização tanto do formato do município como da qualidade das coordenadas geradas.
        Não recebe nenhum parametro e gera png's de todos os os municípios do país na pasta gerada 'plot'.
        Caso queira outro formato, alterar na função plot_coords. """
    
    main_plot()

    """ Tanto country, como state tem funções getters. Caso queria acessar apenas alguns estados ou municípios em específico. """

    #country.get_states(["BA", 13]) #Recebe lista de geocódigos ou siglas de estados

    #state.get_cities([1200013, 1300029, "Rio de Janeiro"]) #Recebe lista de geocódigos ou nomes de municípios



if "__main__" ==  __name__:
    main()
