import numpy as np
import shapely as sh

from multiprocessing import Pool
from time import perf_counter
from os import makedirs, listdir, rename
from os.path import dirname, abspath
from pathlib import Path
from psutil import cpu_count
from gzip import open as gzopen


class city:

    def __init__(self, name:str, geocode:int, coords:np.ndarray) -> None:
        self.name:str = name
        self.geocode:int = geocode
        self.vertices:np.ndarray = coords
    def __str__(self) -> str:
        return ("[%i] %s - vertices: %i"%(self.geocode, self.name, len(self.vertices)))

class state:

    def __init__(self, name:str, sigla:str, geocode:int, cities:list[city], coords:np.ndarray) -> None:
        self.name:str = name
        self.sigla:str = sigla
        self.geocode:int = geocode
        self.cities:list[city] = cities
        self.citiesgc:list[int] = list(c.geocode for c in cities)
        self.vertices:np.ndarray = coords

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

    def __init__(self, name:str, sigla:str, states:list[state], coords:np.ndarray) -> None:
        self.name:str = name
        self.sigla:str = sigla
        self.states:list[state] = states
        self.vertices:np.ndarray = coords

    def add_states(self, states:list[state]) -> None:
        for state in states:
            if (state in self.states):
                print("%s already inclused in %s"%(state.name, self.name))
            else:
                self.states.append(state)

    def get_states(self, names_or_siglas_or_geocodes:list[str|int]) -> list[state]:
        return [st for st in self.states if (st.geocode in names_or_siglas_or_geocodes or st.sigla in names_or_siglas_or_geocodes or st.name in names_or_siglas_or_geocodes)]
    
    def __str__(self) -> str:
        return("%s - %s - vertices: %i\nstates: %i\tcities: %i"%(self.sigla, self.name, len(self.vertices), len(self.states), sum([len(st.cities) for st in self.states])))

def chunck_process(data:bytes) -> list:
    #X[0],Y[1],nome[2],geocodigo[3],vertex_index[4] --- city
    #X[0],Y[1],nome[2],geocodigo[3],vertex_index[4],sigla[5] --- state
    unities:list = []
    lines:list[bytes] = data.splitlines()
    

    i:int = 0
    llen:int = len(lines)
    line:list[bytes] = lines[i].split(b',')
    while (i < llen):
        name:str = line[2].decode('utf-8')
        geocode:int = int(line[3][1:-1])
        vertices:list[list[float]] = [[float(line[0]),float(line[1])]]
        if(len(line)>5): sigla:str = line[5].decode('utf-8')
        i+=1
        line = lines[i].split(b',')
        while (i < llen and int(line[4]) != 0):
            vertices.append([float(line[0]),float(line[1])])
            i+=1
            if (i < llen): line = lines[i].split(b',')
        if(len(line)>5):
            unities.append(state(name, sigla, geocode, [], np.array(vertices)))
        else:
            unities.append(city(name, geocode, np.array(vertices)))
        
    return unities

def cities_gen_p(ct:country, simplify:bool) -> None:
    file_path:Path = Path("%s\\data\\vert_mun_brasil.bin.gz"%(dirname(abspath(__file__))))
    with gzopen(file_path, "rb") as file:
        lines:bytes = file.read()
    
    with Pool(cpu_count()) as p:
        cities_blocks:list[list[city]] = p.map(
            chunck_process, 
            [
            lines[        0: 28955665],
            lines[ 28955665: 58423096],
            lines[ 58423096: 89149001],
            lines[ 89149001:118925057],
            lines[118925057:147005839],
            lines[147005839:176006902],
            lines[176006902:204768193],
            lines[204768193:234440444],
            lines[234440444:263299591],
            lines[263299591:292381024],
            lines[292381024:321690227],
            lines[321690227:         ]
            ]
        )
    
    for block in cities_blocks:
        for cit in block:
            ct.get_states([cit.geocode//100000])[0].add_city(cit)
    
    if (simplify):
        simplify_list:list[int] = [
            1200302,1600402,1600501,1600204,1600279,1301407,1302801,1303536,1304302,1300904,
            1301605,1301308,1303502,1302702,1302207,1303304,1300607,1300144,1300805,1302405,
            1304203,1303908,1302108,1302900,1301951,1304104,1303205,1303007,1301209,1302306,
            1301704,1303809,1300201,1300409,1303601,2917359,5211909,5204409,5218805,5213103,
            5101902,5105101,5106422,5103254,5100805,5108907,5104609,5103502,5101605,5101407,
            5106224,5101704,5102637,5103858,5107180,5107875,5106505,5106240,5107800,5102702,
            5107958,5107065,5107925,5102686,5107859,5105507,5103700,5105150,5103304,5102504,
            5106307,5005400,5006606,5007935,5000203,5002704,5001102,5006903,5007109,5003207,
            1500859,1503002,1505437,1500404,1505031,1503754,1507805,1504208,1506005,1505502,
            1505106,1502764,1507300,1500503,1503606,1500602,1505304,4317103,1100106,1100304,
            1100205,1400308,1400456,1400472,1400282,1400027,1400050,1400209
        ]
        for cit in [ct.get_states([gc//100000])[0].get_cities([gc])[0] for gc in simplify_list]:
            cit.vertices = np.array(sh.simplify(sh.Polygon(cit.vertices), 0.01).exterior.coords, ndmin=2)
    
    for cit in [ct.get_states([int(fv[1:3])])[0].get_cities([int(fv[1:8])])[0] for fv in listdir("%s\\data"%(dirname(abspath(__file__)))) if fv.endswith("vertex_fixed.dat")]:
        cit.vertices = np.loadtxt("%s\\data\\[%i]_%s_vertex_fixed.dat"%(dirname(abspath(__file__)), cit.geocode, cit.name), delimiter=',')

def states_gen_p(ct:country) -> None:
    file_path:Path = Path("%s\\data\\vert_sta_brasil.bin.gz"%(dirname(abspath(__file__))))
    with gzopen(file_path, 'rb', 9) as file:
        lines:bytes = file.read()
    
    with Pool(cpu_count()) as p:
        states_blocks = list(p.imap_unordered(
            chunck_process, 
            [
            lines[       0:  231241],
            lines[  231241:  413755],
            lines[  413755:  696318],
            lines[  696318:  891161],
            lines[  891161: 5276628],
            lines[ 5276628: 6323888],
            lines[ 6323888:10740499],
            lines[10740499:11583136],
            lines[11583136:12818389],
            lines[12818389:13542906],
            lines[13542906:15632880],
            lines[15632880:17423415],
            lines[17423415:17631421],
            lines[17631421:19663022],
            lines[19663022:21453518],
            lines[21453518:22505811],
            lines[22505811:23437304],
            lines[23437304:25072117],
            lines[25072117:25514445],
            lines[25514445:29656463],
            lines[29656463:30902018],
            lines[30902018:32916579],
            lines[32916579:33675800],
            lines[33675800:35161200],
            lines[35161200:36113690],
            lines[36113690:36376432],
            lines[36376432:         ]
            ]
        ))
    
    ct.add_states([st for block in states_blocks for st in block])

def brasil_gen(simplify:bool) -> country:
    Brasil:country = country("Brasil", "BR", [], np.array([]))
    start = perf_counter()
    states_gen_p(Brasil)
    cities_gen_p(Brasil, simplify)
    print(perf_counter()- start)

    return Brasil



def Hex_Points_in_Polygon_forced(polygon:sh.Polygon, r:float) -> np.ndarray:
    xys:np.ndarray = (np.column_stack((2*r*0.01*np.round(np.cos(np.arange(0, 2*np.pi, np.pi/3)), 6),
                                      2*r*0.01*np.round(np.sin(np.arange(0, 2*np.pi, np.pi/3)), 6))))
    points:list[list[float]] = [[polygon.centroid.x, polygon.centroid.y]]
    rect:sh.Polygon = sh.minimum_rotated_rectangle(polygon)
    sh.prepare(rect)
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
    
    points_array:np.ndarray = np.array(points)
    return points_array[np.vectorize(sh.within, signature='(a),()->(a)')(np.vectorize(sh.Point,signature='(a)->()')(points_array), polygon)]

def city_based_coords_gen(st:state, cit:city, r:float) -> int:
    
    cit_shape:sh.Polygon = sh.Polygon(cit.vertices)

    xys = Hex_Points_in_Polygon_forced(cit_shape, r)
    if (xys.shape[0] < 0.8*cit_shape.area*10000//(np.pi*r**2)):
        print("%s[%i] poucas coordenadas: %i -> esperado: %i\tForça bruta falhou..."%(cit.name, cit.geocode, xys.shape[0], cit_shape.area*10000//(np.pi*r**2)))
    
    
    makedirs("%s\\outputs\\coords\\%s"%(dirname(abspath(__file__)), st.sigla), exist_ok=True)
    np.savetxt("%s\\outputs\\coords\\%s\\[%i]_%s_coords.dat"%(dirname(abspath(__file__)), st.sigla, cit.geocode, cit.name), xys, "%.13f",',')

    return xys.shape[0]

def plot_coords(st:state, cit:city|None=None) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import use
    use("Agg")

    makedirs("%s\\outputs\\plots\\%s"%(dirname(abspath(__file__)), st.sigla), exist_ok=True)

    br:str = next(f for f in listdir("%s\\outputs"%(dirname(abspath(__file__)))) if f.startswith("Brasil"))
    stfolder:str = next(f for f in listdir("%s\\outputs\\%s"%(dirname(abspath(__file__)), br)) if f[0:2] == st.sigla)
    if(cit):
        xys:np.ndarray = np.loadtxt("%s\\outputs\\%s\\%s\\[%i]_%s_coords.dat"%(dirname(abspath(__file__)), br, stfolder, cit.geocode, cit.name), delimiter=',', ndmin=2)
        plt.plot(cit.vertices[:,0], cit.vertices[:,1], scalex=True, scaley=True, color="#DB5C1F")
        plt.scatter(xys[:, 0], xys[:, 1],color="red",s=0.5)
        plt.title("[%i] %s - %s"%(cit.geocode, cit.name, st.sigla))
        plt.savefig("%s\\outputs\\plots\\%s\\[%i]_%s.png"%(dirname(abspath(__file__)), st.sigla, cit.geocode, cit.name), backend='Agg', dpi=200)
    else:
        xys = np.concatenate(
            [np.loadtxt("%s\\outputs\\%s\\%s\\%s"%(dirname(abspath(__file__)), br, stfolder, f), delimiter=',', ndmin=2) for f in listdir("%s\\outputs\\%s\\%s"%(dirname(abspath(__file__)), br, stfolder))]
        )
        plt.plot(st.vertices[:,0], st.vertices[:,1], scalex=True, scaley=True, color="#DB5C1F")
        plt.scatter(xys[:, 0], xys[:, 1],color="red",s=0.5)
        plt.title("[%i] %s - %s"%(st.geocode, st.name, st.sigla))
        plt.savefig("%s\\outputs\\plots\\%s\\[%i]_%s.png"%(dirname(abspath(__file__)), st.sigla, st.geocode, st.name), backend='Agg', dpi=200)

    plt.close()



def main_gen(r:float) -> None:
    Brasil:country = brasil_gen(True)

    start:float = perf_counter()
    total_coords:int = 0
    with Pool(cpu_count()) as p:
        for st in Brasil.states:
            start_time:float = perf_counter()
            state_coords:int = sum(p.starmap(city_based_coords_gen, [[st, cit, r] for cit in st.cities]))

            rename("%s\\outputs\\coords\\%s"%(dirname(abspath(__file__)), st.sigla), "%s\\outputs\\coords\\%s[%i]"%(dirname(abspath(__file__)), st.sigla, state_coords))

            total_coords += state_coords
            print("\nTotal de coordenadas %s: %i"%(st.sigla, state_coords))
            print("execution time %s: %0.6f\n"%(st.sigla, perf_counter()-start_time))

    rename("%s\\outputs\\coords"%(dirname(abspath(__file__))), "%s\\outputs\\Brasil[%i]_coordinates"%(dirname(abspath(__file__)), total_coords))

    print("Total de coordenadas %s: %i"%(Brasil.name, total_coords))
    print("execution time:", perf_counter()-start)

def main_plot() -> None:
    Brasil:country = brasil_gen(False)

    start_time:float = perf_counter()
    with Pool(cpu_count()) as p:
        for st in Brasil.states:
            start_st:float = perf_counter()
            plot_coords(st)
            p.starmap(plot_coords, [[st, cit] for cit in st.cities], (len(st.cities)//cpu_count())+1)
            print("plot duration %s: %0.6f\n"%(st.sigla, perf_counter()-start_st))

    print("general plot duration: %f"%(perf_counter()-start_time))

def main() -> None:
    """ Gerador de coordenadas. Recebe apenas o raio em Km.
        Gerará pastas para todo os estados dentro da pasta coords. Depois só o nome da pasta para começar com Brasil.
        Caso queira que as pastas gere o número de coordenadas, basta descomentar os renames na função main_gen. 
        Porém lembrar de comentá-los ou mudar o nome da pasta coords para não gerar erros em próximas execuções. """
    
    #main_gen(1.35)

    """ Gerador de gráficos pra visualização tanto do formato do município como da qualidade das coordenadas geradas.
        Não recebe nenhum parametro e gera png's de todos os os municípios do país na pasta gerada 'plot'.
        Caso queira outro formato, alterar na função plot_coords. """
    
    main_plot()

    """ Tanto country, como state tem funções getters. Caso queria acessar apenas alguns estados ou municípios em específico. """

    #country.get_states(["BA", 13]) #Recebe lista de geocódigos ou siglas de estados

    #state.get_cities([1200013, 1300029, "Rio de Janeiro"]) #Recebe lista de geocódigos ou nomes de municípios



if "__main__" ==  __name__:
    main()
