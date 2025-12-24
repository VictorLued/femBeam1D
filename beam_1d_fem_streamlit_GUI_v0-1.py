#---------------
# Imports
# -------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

#---------------
# Options for better readability of printed arrays
# -------------
np.set_printoptions(
    linewidth=120,    # maximum line width when printing arrays
    precision=3,      # number of decimals displayed
    suppress=True,    # do not use scientific notation for small numbers
    edgeitems=5       # number of elements shown at the beginning/end of large arrays
)


def get_Second_Moment_of_Area(profile, H, h, B, b):
    """
    Calculate the second moment of area (Flächenträgheitsmoment I)
    for different profiles.

    Parameters
    ----------
    profile : int
        1 = Rechteckprofil (rectangular hollow section)
        2 = I-Träger (I-beam)
        3 = C-Träger (C-beam)
        4 = Kreisprofil (circular hollow section)
    H : float
    h : float
    B : float
    b : float

    Returns
    -------
    I : float
        Second moment of area about the centroidal x-axis (Ix)
    z : float
        Centroidal axis location (always H/2 for symmetric sections)
    """

    if profile == 1:  # Rechteck-Hohlprofil
        # Inner dimensions
        I = (B * H**3 - b * h**3) / 12
        z = H / 2
        return I, z

    elif profile == 2:  # I-Träger (simplified model)
        b2 = B - b
        h2 = H - 2*h
        I = (B * H**3 - b2 * h2**3) / 12
        z = H / 2
        return I, z

    elif profile == 3:  # C-Träger (simplified model)
        b2 = B - b
        h2 = H - 2*h
        I = (B * H**3 - b2 * h2**3) / 12
        z = H / 2
        return I, z

    elif profile == 4:  # Kreis-Hohlprofil
        # H = outer diameter, h = inner diameter
        I = (np.pi / 64) * (H**4 - h**4)
        z = H / 2
        return I, z

    else:
        raise ValueError("Unknown profile type. Must be 1, 2, 3, or 4.")


def generate_Mesh(modelData):

    modelData = modelData.sort_values("x [mm]").reset_index(drop=True)
    
    positions = modelData["x [mm]"]
    nodes = modelData["Node"]

    elementIds = []
    elementNode1 = []
    elementNode2 = []
    elementPos1 = []
    elementPos2 = []
    elementLengths = []
    
    for index, currentNode in enumerate(nodes[:-1]):
        elementIds.append(index)
        elementNode1.append(nodes[index])
        elementNode2.append(nodes[index+1])
        elementPos1.append(positions[index])
        elementPos2.append(positions[index+1])
        elementLengths.append(positions[index+1]-positions[index])
    
    mesh = pd.DataFrame({
        "Element Id": elementIds,
        "Node 1": elementNode1,
        "Node 2": elementNode2,
        "Position 1": elementPos1,
        "Position 2": elementPos2,
        "Length": elementLengths
    })

    return mesh


def get_Global_Force(modelData):
    forces = modelData["F [N]"].to_numpy(dtype=float)
    moments = modelData["M [Nmm]"].to_numpy(dtype=float)
    nodeIds = modelData["Node"].to_numpy(dtype=int)

    numberOfNodes = nodeIds.shape[0]  
    F = np.zeros((2*numberOfNodes, 1)) 

    # Vertikale Kräfte auf gerade DOFs
    F[2*nodeIds, 0] = forces

    # Momente auf ungerade DOFs
    F[2*nodeIds + 1, 0] = moments

    return F


def get_Beam_2DStiffness(E, I, Le):
    """
    Compute the local stiffness matrix for a 2-node Euler-Bernoulli beam element
    E  : Young's modulus
    I  : second moment of area
    Le : element length
    """
    ke = (E*I)/Le**3*np.array(
        [[12, 6*Le, -12, 6*Le],
         [6*Le, 4*Le**2, -6*Le, 2*Le**2],
         [-12, -6*Le, 12, -6*Le],
         [6*Le, 2*Le**2, -6*Le, 4*Le**2]])
    return ke
     

def get_Global_Stiffness(E, I, mesh):
    """
    Assemble the global stiffness matrix for the entire beam
    E               : Young's modulus
    I               : second moment of area
    L               : total beam length
    numberOfElements: number of finite elements
    Returns:
        K              : global stiffness matrix
        nodesIds       : array of node indices
        xCoordinates   : array of node positions
        elementIds     : array of element indices
        elementLengths : array of element lengths
    """
    numberOfElements = mesh.shape[0]
    numberOfNodes = mesh.shape[0]+1
    
    K = np.zeros([2*numberOfNodes, 2*numberOfNodes])    # initialize global stiffness matrix

    # Assemble global stiffness matrix element by element
    for elementIndex in mesh["Element Id"]:
        # Determine global DOF indices for the current element
        elementDOF = [2*mesh["Node 1"][elementIndex], 2*mesh["Node 1"][elementIndex]+1, 2*mesh["Node 2"][elementIndex], 2*mesh["Node 2"][elementIndex]+1]
        Le =  mesh["Length"][elementIndex]
        ke = get_Beam_2DStiffness(E, I, Le)
        # Add local element stiffness to global stiffness matrix
        K[np.ix_(elementDOF, elementDOF)] += ke
    return K


def include_Boundaries(modelData, K, F):
    # 1. Spring Stiffnesses
    for index, nodeId in enumerate(modelData["Node"]):
        apply_Boundaries_Spring(nodeId, modelData["k [N/mm]"][index], modelData["kr [Nmm/-]"][index], K)

    # 2 Displacement Boundaries
    delDOFs, K, F = apply_Boundaries_Displacement(modelData, K, F)

    return delDOFs, K, F


def apply_Boundaries_Spring(nodeId, k1, k2, K):
    """
    Apply boundary conditions using spring stiffness method (supports elastic constraints)
    nodeId : index of the node where the BC is applied
    k1     : translational spring stiffness (vertical displacement)
    k2     : rotational spring stiffness (rotation)
    K      : global stiffness matrix (modified in place)
    """
    K[2*nodeId, 2*nodeId] += k1
    K[2*nodeId+1, 2*nodeId+1] += k2
    

def apply_Boundaries_Displacement(modelData, K, F):
    delDOFs = []

    # Disp != 0
    prescribed_DOFs = []
    prescribed_Value = []
    
    # Prescribed Displacement: alles was NICHT "free" ist und ≠ 0
    mask_disp = (modelData["Disp [mm]"] != "free") & (modelData["Disp [mm]"] != 0)
    prescribed_disp = 2 * (modelData.loc[mask_disp, "Node"])
    prescribed_Value.extend(modelData["Disp [mm]"][mask_disp])
    prescribed_DOFs.extend(prescribed_disp.tolist())
    
    # Prescribed Rotation: alles was NICHT "free" ist und ≠ 0
    mask_rot = (modelData["Rot [-]"] != "free") & (modelData["Rot [-]"] != 0)
    precribed_rot = 2 * (modelData.loc[mask_rot, "Node"]) + 1
    prescribed_Value.extend(modelData["Rot [-]"][mask_rot])
    prescribed_DOFs.extend(precribed_rot.tolist())

    for index, dof in enumerate(prescribed_DOFs):
        K[dof, :] = 0      # Null out row
        K[dof, dof] = 1    # Set diagonal to 1
        F[dof, 0] = prescribed_Value[index]

    
    # Disp = 0
    fixed_disp = 2 * modelData["Node"][modelData["Disp [mm]"] == 0] 
    delDOFs.extend(fixed_disp.tolist())
    fixed_rot = 2 * modelData["Node"][modelData["Rot [-]"] == 0] + 1
    delDOFs.extend(fixed_rot.tolist())

    delDOFs = sorted(set(map(int, delDOFs)))

    # K und F anpassen
    K = np.delete(K, delDOFs, axis=0)
    K = np.delete(K, delDOFs, axis=1)
    F = np.delete(F, delDOFs, axis=0)

    return delDOFs, K, F

def solve(K, F):
    """
    Solve the linear system Ku = F for nodal displacements u
    K : global stiffness matrix
    F : global force vector
    Returns:
        u : nodal displacement vector
    """
    u = np.linalg.inv(K) @ F
    return u


def get_Global_Displacement(u, delDOFs, mesh):
    """
    Baut aus dem reduzierten Lösungsvektor u wieder den vollständigen Verschiebungsvektor.
    
    u       : reduzierte Lösung (ohne gelöschte DOFs)
    delDOFs : Liste der entfernten DOFs (z.B. wegen Randbedingungen)
    mesh    : Elementmesh (wird genutzt um Anzahl der Knoten zu bestimmen)
    """
    numberOfNodes = mesh.shape[0]
    disp = np.zeros((2*numberOfNodes, 1))

    # Indizes der verbliebenen DOFs
    allDOFs = np.arange(2*numberOfNodes)
    keptDOFs = np.delete(allDOFs, delDOFs)

    # reduzierten Vektor u in die freien DOFs einfügen
    disp[keptDOFs, 0] = u.flatten()

    return disp


# Hermite shape functions for a 2-node beam element (cubic interpolation)
def N1(x, L):
    """Shape function associated with displacement at node 0"""
    return 1 - 3*(x**2)/(L**2) + 2*(x**3)/(L**3)

def N2(x, L):
    """Shape function associated with rotation at node 0"""
    return x - 2*(x**2)/L + (x**3)/(L**2)

def N3(x, L):
    """Shape function associated with displacement at node 1"""
    return 3*(x**2)/(L**2) - 2*(x**3)/(L**3)

def N4(x, L):
    """Shape function associated with rotation at node 1"""
    return -(x**2)/L + (x**3)/(L**2)

def get_Local_Element_Disp(x, dofs, L):
    """
    Compute the displacement within an element using Hermite shape functions
    x    : local coordinate along the element [0, L]
    dofs : element degrees of freedom [w0, phi0, wL, phiL]
    L    : element length
    """
    w0, phi0, wL, phiL = dofs
    w = N1(x, L) * w0 + N2(x, L) * phi0 + N3(x, L) * wL +  N4(x, L) * phiL  
    return w

def get_Local_Element_Stress(x, dofs, L, E, z):
    kappa = get_Element_Curvature(x, dofs, L)
    
    sigma_top = -E * z * kappa
    sigma_bottom = E * z * kappa
    
    return sigma_top, sigma_bottom

def N1_dd(x,L):
    return -6/L**2 + 12*x/L**3

def N2_dd(x,L):
    return -4/L + 6*x/L**2

def N3_dd(x,L):
    return 6/L**2 - 12*x/L**3

def N4_dd(x,L):
    return -2/L + 6*x/L**2

def get_Element_Curvature(x, dofs, L):
    w0, theta0, wL, thetaL = dofs
    kappa = N1_dd(x,L)*w0 + N2_dd(x,L)*theta0 + N3_dd(x,L)*wL + N4_dd(x,L)*thetaL
    return kappa

# -------------------------
# Umwandlung für FEM-Logik
# -------------------------
def process_model_data(df):
    df_fem = df.copy()

    # Freiheitsgrade: fehlende Displacement/Rotation = "free"
    for col in ["Disp [mm]", "Rot [-]"]:
        df_fem[col] = df_fem[col].apply(lambda x: "free" if pd.isna(x) else float(x))

    # Andere Werte: fehlend = 0, ansonsten float
    for col in ["F [N]", "M [Nmm]", "k [N/mm]", "kr [Nmm/-]"]:
        df_fem[col] = df_fem[col].apply(lambda x: 0.0 if pd.isna(x) else float(x))

    return df_fem


def plot_Beam_Model(disp, modelData, mesh, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    xCoordinates = modelData["x [mm]"].to_numpy()
    x_span = np.max(xCoordinates) - np.min(xCoordinates)

    y_offset_elem = 0.015 * x_span 
    y_offset_node = 0.015 * x_span  
    x_offset = 0.01 * x_span
    symbol_offset = -0.015 * x_span

    # Flags, damit Legenden-Eintrag nur einmal erzeugt wird
    legend_flags = {
        "disp": True,
        "force": True,
        "moment": True,
        "rot": True,
        "spring": True,
        "rotspring": True
    }

    # -------------------------
    # Randbedingungen und Lasten
    # -------------------------
    for _, row in modelData.iterrows():
        x = row["x [mm]"]

        # Verschiebung[-]
        disp_val = row["Disp [mm]"]
        if disp_val == 0:
            ax.plot(x, symbol_offset, "^", color="blue", ms=15,
                    label="Verschiebung" if legend_flags["disp"] else None)
            legend_flags["disp"] = False
        elif disp_val not in (0, None, "free"):
            dy = 0.1 * x_span * np.sign(disp_val)
            ax.annotate('', xy=(x, dy), xytext=(x, 0),
                        arrowprops=dict(facecolor='blue', edgecolor='blue',
                                        width=2, headwidth=8, headlength=10),
                        label="Verschiebung" if legend_flags["disp"] else None)
            legend_flags["disp"] = False

        # Kraft
        Fval = row["F [N]"]
        if Fval not in (0, None):
            dy = 0.1 * x_span * np.sign(Fval)
            ax.annotate('', xy=(x, dy), xytext=(x, 0),
                        arrowprops=dict(facecolor='red', edgecolor='red',
                                        width=2, headwidth=8, headlength=10),
                        label="Kraft" if legend_flags["force"] else None)
            legend_flags["force"] = False

        # Moment
        Mval = row["M [Nmm]"]
        if Mval not in (0, None):
            radius = 0.05 * x_span
            theta = np.linspace(0, np.pi/2, 20)
            if Mval > 0:
                x_arc = x + radius * np.cos(theta)
                y_arc = radius * np.sin(theta)
            else:
                x_arc = x + radius * np.cos(theta)
                y_arc = -radius * np.sin(theta)
            ax.plot(x_arc[:-1], y_arc[:-1], color='red', lw=2,
                    label="Moment" if legend_flags["moment"] else None)
            ax.annotate('', xy=(x_arc[-1], y_arc[-1]), xytext=(x_arc[-2], y_arc[-2]),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))
            legend_flags["moment"] = False

        # Rotation
        rot_val = row["Rot [-]"]
        if rot_val == 0:
            ax.plot(x, 0, 'x', color="blue", ms=15,
                    label="Rotation" if legend_flags["rot"] else None)
            legend_flags["rot"] = False
        elif rot_val not in (0, None, "free"):
            radius = 0.05 * x_span
            theta = np.linspace(0, np.pi/2, 20)
            if rot_val < 0:
                x_arc = x + radius * np.cos(theta)
                y_arc = -radius * np.sin(theta)
            else:
                x_arc = x + radius * np.cos(theta)
                y_arc = radius * np.sin(theta)
            ax.plot(x_arc[:-1], y_arc[:-1], color='blue', lw=2,
                    label="Rotation" if legend_flags["rot"] else None)
            ax.annotate('', xy=(x_arc[-1], y_arc[-1]), xytext=(x_arc[-2], y_arc[-2]),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))
            legend_flags["rot"] = False

        # Feder k
        k_val = row["k [N/mm]"]
        if k_val not in (0, None):
            ax.text(x, symbol_offset, "z", color="purple", fontsize=25,
                    ha='center', va='center')
            if legend_flags["spring"]:
                ax.plot([], [], color="purple", marker="$z$", linestyle="None", label="Feder k")
                legend_flags["spring"] = False

        # Rotationsfeder kr
        kr_val = row["kr [Nmm/-]"]
        if kr_val not in (0, None):
            ax.plot(x, 0, 'x', color="purple", ms=15,
                    label="Rotationsfeder" if legend_flags["rotspring"] else None)
            legend_flags["rotspring"] = False

    # -------------------------
    # Elemente zeichnen (blau)
    # -------------------------
    first_elem = True
    for _, elem in mesh.iterrows():
        x1, x2 = elem["Position 1"], elem["Position 2"]
        label = "Elements" if first_elem else None
        ax.plot([x1, x2], [0, 0], "-b", lw=2, label=label)
        first_elem = False

        # Element-ID
        xm = 0.5 * (x1 + x2)
        ax.text(xm, -y_offset_elem, str(int(elem["Element Id"])), color="blue",
                ha="center", va="top", fontsize=10)

    # -------------------------
    # Knoten zeichnen (schwarz)
    # -------------------------
    ax.plot(xCoordinates, np.zeros_like(xCoordinates), "o", color="black", label="Nodes")
    for _, node in modelData.iterrows():
        ax.text(node["x [mm]"]+x_offset, y_offset_node, str(int(node["Node"])), color="black",
                ha="center", va="bottom", fontsize=10)

    # -------------------------
    # Layout
    # -------------------------
    ax.set_xlabel("x in mm")
    ax.set_ylabel("Displacement in mm")
    ax.set_title("Beam Model")
    ax.axis("equal")
    ax.grid(True)
    #ax.legend()

    return ax




def plot_Beam_Disp(disp, modelData, mesh, ax=None, npts=300, cmap="jet", proportional=True):
    if ax is None:
        ax = plt.gca()

    X, Y, C = [], [], []

    for _, row in mesh.iterrows():
        node1, node2 = int(row["Node 1"]), int(row["Node 2"])
        L = float(row["Length"])
        x0, x1 = float(row["Position 1"]), float(row["Position 2"])

        xg = np.linspace(x0, x1, npts)
        xl = xg - x0

        dofs = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
        ue = disp[dofs, 0]

        w = get_Local_Element_Disp(xl, ue, L)

        X.extend(xg)
        Y.extend(w)
        C.extend(w)

    sc = ax.scatter(X, Y, c=C, cmap=cmap, s=5)

    # Knotenpunkte
    nodeX = modelData["x [mm]"].to_numpy().astype(float)
    nodeY = disp[0::2, 0].astype(float)
    ax.plot(nodeX, nodeY, "o", color="black", ms=6, label="Nodes")

    # Achsenbegrenzung
    x_min, x_max = np.min(nodeX), np.max(nodeX)
    if proportional:
        y_limit = 0.4 * (x_max - x_min)
        ax.set_xlim(x_min, 1.05*x_max)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.autoscale(enable=True, axis='both')

    ax.set_xlabel("x in mm")
    ax.set_ylabel("Displacement in mm")
    ax.set_title("Beam Displacement")
    ax.grid(True)
    ax.legend()

    fig = ax.figure
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Displacement [mm]")

    return ax, sc


def plot_Beam_Stress(disp, modelData, mesh, E, z, side="mises", ax=None, npts=300, cmap="jet", proportional=True):
    if ax is None:
        ax = plt.gca()

    X, Y, C = [], [], []

    for _, row in mesh.iterrows():
        node1, node2 = int(row["Node 1"]), int(row["Node 2"])
        L = float(row["Length"])
        x0, x1 = float(row["Position 1"]), float(row["Position 2"])

        xg = np.linspace(x0, x1, npts)
        xl = xg - x0

        dofs = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
        ue = disp[dofs, 0]

        w = get_Local_Element_Disp(xl, ue, L)

        for xi, xi_local, wi in zip(xg, xl, w):
            sigma_top, sigma_bottom = get_Local_Element_Stress(xi_local, ue, L, E, z)
            if side == "mises":
                sigma = abs(sigma_top)
                stressTitle = f"von Mises Stress Distribution"
            elif side == "top":
                sigma = sigma_top
                stressTitle = f"Normal Stress - {side} fiber"
            else:
                sigma = sigma_bottom
                stressTitle = f"Normal Stress - {side} fiber"
            X.append(xi)
            Y.append(wi)
            C.append(sigma)

    sc = ax.scatter(X, Y, c=C, cmap=cmap, s=5)

    # Knotenpunkte
    nodeX = modelData["x [mm]"].to_numpy().astype(float)
    nodeY = disp[0::2, 0].astype(float)
    ax.plot(nodeX, nodeY, "o", color="black", ms=6, label="Nodes")

    # Achsenbegrenzung
    x_min, x_max = np.min(nodeX), np.max(nodeX)
    if proportional:
        y_limit = 0.4 * (x_max - x_min)
        ax.set_xlim(x_min, 1.05*x_max)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.autoscale(enable=True, axis='both')

    ax.set_xlabel("x in mm")
    ax.set_ylabel("Displacement in mm")
    ax.set_title(stressTitle)
    ax.grid(True)
    ax.legend()

    fig = ax.figure
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Stress [N/mm²]")

    return ax, sc, np.array(C)



# -------------------------
# MAIN
# -------------------------
st.set_page_config(
    page_title="1D Beam FEM Simulator",
    layout="wide",  # <--- "wide" statt "centered"
    initial_sidebar_state="auto"
)


st.title("1D Beam FEM Simulator (v0.1)")
st.page_link("https://www.linkedin.com/in/victor-l%C3%BCddemann-309559208/", label="Made by me! - Victor Lüddemann", icon=":material/person:")

# -------------------------
# Session-State Initialisierung
# -------------------------
if "model_built" not in st.session_state:
    st.session_state.model_built = False
if "solved" not in st.session_state:
    st.session_state.solved = False
if "dirty" not in st.session_state:
    st.session_state.dirty = True

# -------------------------
# Eingabeparameter
# -------------------------
st.sidebar.header("Simulation Parameters")
number_Elements = st.sidebar.slider("Number of Elements", 1, 10, 2)
number_Nodes = number_Elements + 1

E = st.sidebar.number_input("Youngs Modulus [N/mm²]", value=210000.0, step=1000.0)
L = st.sidebar.number_input("Length [mm]", value=10000.0, step=5.0)

profileType = st.sidebar.selectbox(
    "Profile Type", 
    [1, 2, 3, 4, 5], 
    index=0, 
    format_func=lambda x: {1: "Rectangle", 2: "I-Profile", 3: "C-Profile", 4: "Circle", 5: "Custom"}[x]
)

# Geometrie-/Querschnittswerte
if profileType in [1, 2, 3, 4]:
    # Nur für Standardprofile die Geometrie abfragen
    H = st.sidebar.number_input("H [mm]", value=100.0, step=5.0)
    h = st.sidebar.number_input("h [mm]", value=0.0, step=1.0)
    B = st.sidebar.number_input("B [mm]", value=50.0, step=5.0)
    b = st.sidebar.number_input("b [mm]", value=0.0, step=1.0)

    I, z = get_Second_Moment_of_Area(profileType, H, h, B, b)
    W = I / z if z != 0 else 0.0

else:
    # Custom-Profil: I und W direkt eingeben
    I = st.sidebar.number_input(
        "Area moment of inertia $I_y$ [mm⁴]",
        value=1.0e7,
        step=1.0e6,
        format="%.2f"
    )
    W = st.sidebar.number_input(
        "Section modulus $W_y$ [mm³]",
        value=1.0e5,
        step=1.0e4,
        format="%.2f"
    )

    z = I / W if W != 0 else 0.0

st.sidebar.write(f"Area moment of inertia $I_y$ = {I:.2f} mm⁴")
st.sidebar.write(f"Section Modulus $W_y$ = {W:.2f} mm³")
st.sidebar.write("\n\n*Remark: Use this solver with caution - No liability is assumed.*")

# -------------------------
# Bilder der Profiltypen anzeigen
# -------------------------
st.markdown("### Profile Reference")
if profileType == 1:
    st.image("images/profile_rect.png", caption="Profile Type: Rectangle", width=150)
elif profileType == 2:
    st.image("images/profile_I.png", caption="Profile Type: I Profile", width=150)
elif profileType == 3:
    st.image("images/profile_c.png", caption="Profile Type: C Profil", width=150)
elif profileType == 4:
    st.image("images/profile_circle.png", caption="Profile Type: Circle", width=180)
elif profileType == 5:
    st.info("Custom profile: using user-defined $I_y$ and $W_y$.")

# -------------------------
# Knoten- und Lasttabelle
# -------------------------

tableHeader = [
    "Node", "x [mm]", "Disp [mm]", "Rot [-]",
    "F [N]", "M [Nmm]", "k [N/mm]", "kr [Nmm/-]"
]

# Initialisiere DataFrame
node_ids = np.arange(number_Nodes)  # Knoten = Elemente + 1
x_coords = np.linspace(0, L, number_Nodes)

data = {
    "Node": node_ids,
    "x [mm]": x_coords,
    "Disp [mm]": [0.0] + [None]*(number_Nodes - 2) + [None],
    "Rot [-]": [0.0] + [None]*(number_Nodes - 2) + [None],
    "F [N]": [None]*(number_Nodes - 1) + [-1000.0],
    "M [Nmm]": [None]*(number_Nodes),
    "k [N/mm]": [None]*(number_Nodes),
    "kr [Nmm/-]": [None]*(number_Nodes)
}

df_nodes = pd.DataFrame(data)


# -------------------------
# Darstellung
# -------------------------

st.subheader("Node Definition")

# DataEditor anzeigen
edited_df = st.data_editor(
    df_nodes,
    column_config={
        "Node": st.column_config.TextColumn("Node", disabled=True),
    },
    hide_index=True
)

# Falls der Nutzer den Editor verändert hat
modelData = process_model_data(edited_df)

csv_data_model = modelData.to_csv(index=True, sep=';', encoding='utf-8')

st.markdown("""
- **Disp [mm]** – Prescribed displacement (boundary condition)
- **Rot [-]** – Prescribed rotation (boundary condition)
- **F [N]** – Nodal force (transverse load)
- **M [Nmm]** – Nodal bending moment
- **k [N/mm]** – Translational spring stiffness
- **kr [Nmm/-]** – Rotational spring stiffness
""")

# -------------------------
# Prüfen ob Eingaben verändert wurden -> dirty setzen
# -------------------------
if (
    number_Elements != st.session_state.get("last_number_Elements") or
    E != st.session_state.get("last_E") or
    L != st.session_state.get("last_L") or
    not edited_df.equals(st.session_state.get("last_df", edited_df))
):
    st.session_state.dirty = True
    st.session_state.model_built = False
    st.session_state.solved = False

# Eingaben für nächsten Vergleich merken
st.session_state.last_number_Elements = number_Elements
st.session_state.last_E = E
st.session_state.last_L = L
st.session_state.last_df = edited_df.copy()

# -------------------------
# Build Model
# -------------------------
st.subheader("Model")
if st.button("Build Model", icon = ":material/settings:"):
    mesh = generate_Mesh(modelData)
    K = get_Global_Stiffness(E, I, mesh)
    F = get_Global_Force(modelData)
    delDOFs, K_red, F_red = include_Boundaries(modelData, K, F)

    # Ergebnisse in Session speichern
    st.session_state.modelData = modelData
    st.session_state.mesh = mesh
    st.session_state.K_red = K_red
    st.session_state.F_red = F_red
    st.session_state.delDOFs = delDOFs

    # Flags setzen
    st.session_state.model_built = True
    st.session_state.solved = False
    st.session_state.dirty = False

# -------------------------
# Ergebnisse anzeigen
# -------------------------
if st.session_state.model_built:
    # Plot + Matrizen anzeigen (bleiben immer sichtbar)
    fig, ax = plt.subplots(figsize=(10,4))
    plot_Beam_Model(np.zeros_like(st.session_state.F_red), modelData, st.session_state.mesh, ax=ax)
    st.pyplot(fig)


    # -------------------------
    # DOFs berechnen (reduziert)
    # -------------------------
    n_total = st.session_state.K_red.shape[0] + len(st.session_state.delDOFs)
    all_dofs = list(range(n_total))
    red_dofs = [d for d in all_dofs if d not in st.session_state.delDOFs]

    # Create dataframes
    df_model_Kred = pd.DataFrame(st.session_state.K_red, index=red_dofs, columns=red_dofs)
    csv_data_K = df_model_Kred.to_csv(index=True, sep=';', encoding='utf-8')

    df_model_Fred = pd.DataFrame(st.session_state.F_red, index=red_dofs, columns=[""])
    csv_data_F = df_model_Fred.to_csv(index=True, sep=';', encoding='utf-8')

    # Create a styled version for display
    styled_Kred = df_model_Kred.style.format("{:.2e}")
    styled_Fred = df_model_Fred.style.format("{:.2e}")


    st.subheader("Matrix")

    matrix_col1, matrix_col2, matrix_col3 = st.columns([6, 1.5, 0.5])

    with matrix_col1:
        st.markdown("**Global Stiffness Matrix K**")
        st.dataframe(styled_Kred)

    with matrix_col2:
        st.markdown("**Load Vector F**")
        st.dataframe(styled_Fred)


    # -------------------------
    # Solve nur anzeigen, wenn Model gebaut wurde
    # -------------------------
    if st.button("Solve Model", icon=":material/play_circle:"):
        mesh = st.session_state.mesh
        K_red = st.session_state.K_red
        F_red = st.session_state.F_red
        delDOFs = st.session_state.delDOFs

        u_red = solve(K_red, F_red)
        u = get_Global_Displacement(u_red, delDOFs, modelData)
        st.session_state.solution = (u_red, u, mesh)

        # solved-Flag setzen
        st.session_state.solved = True

# -------------------------
# Displacement & Stress Plots (nur wenn solved)
# -------------------------
if st.session_state.solved:
    prop = st.checkbox("True Scale", value=False)

    u_red, u, mesh = st.session_state.solution

    df_model_Ured = pd.DataFrame(u_red, index=red_dofs, columns=[""])
    styled_Ured = df_model_Ured.style.format("{:.2e}")
    csv_data_u = df_model_Ured.to_csv(index=True, sep=';', encoding='utf-8')

    st.subheader("Beam Displacement")

    solution_col1, solution_col2, solution_col3 = st.columns([5, 2, 0.5])

    with solution_col1:
        fig, ax = plt.subplots(figsize=(10,4))
        plot_Beam_Disp(u, modelData, mesh, ax=ax, npts=1000, proportional=prop)
        st.pyplot(fig)

        # Maximalverschiebung berechnen
        max_disp = np.max(np.abs(u))
        st.markdown(f"**Maximum Displacement:** {max_disp:.3e} mm")

    with solution_col2:
        st.markdown("**Displacement Vector u**")
        st.dataframe(styled_Ured)



    st.subheader("von Mises Stress")
    col1, col2, col3 = st.columns([0.1, 5, 0.1])
    with col2:
        fig1, ax1= plt.subplots(figsize=(10,4))
        _, _, stress_Mises = plot_Beam_Stress(u, modelData, mesh, E, z, side="mises", ax=ax1, npts=1000, proportional=prop)
        st.pyplot(fig1)

    # Extremwerte der Spannungen berechnen
    sigma_mises_max = np.max(stress_Mises)
    st.markdown(f"**Maximum von Mises Stress:** {sigma_mises_max:.3e} MPa")


    st.subheader("Normal Stress")
    col1, col2 = st.columns([4, 4])
    with col1:
        fig2, ax2 = plt.subplots(figsize=(10,4))
        _, _, stress_top = plot_Beam_Stress(u, modelData, mesh, E, z, side="top", ax=ax2, npts=1000, proportional=prop)
        st.pyplot(fig2)

    with col2:
        fig3, ax3 = plt.subplots(figsize=(10,4))
        _, _, stress_bottom = plot_Beam_Stress(u, modelData, mesh, E, z, side="bottom", ax=ax3, npts=1000, proportional=prop)
        st.pyplot(fig3)

    # Extremwerte der Spannungen berechnen
    all_stresses = np.concatenate([stress_top, stress_bottom])
    sigma_max = np.max(all_stresses)
    sigma_min = np.min(all_stresses)

    st.markdown(f"**Maximum Bending Stress:** {sigma_max:.3e} MPa ;  **Minimum Bending Stress:** {sigma_min:.3e} MPa")

    # -------------------------
    # DataFrame für Export
    # -------------------------

    # Daten sammeln
    x_vals = np.linspace(modelData["x [mm]"].min(), modelData["x [mm]"].max(), 100)

    disp_list = []
    top_stress_list = []
    bottom_stress_list = []
    mises_stress_list = []

    for xi in x_vals:
        # Lokales Element und DOFs bestimmen
        row = mesh[(mesh["Position 1"] <= xi) & (mesh["Position 2"] >= xi)].iloc[0]
        node1, node2 = int(row["Node 1"]), int(row["Node 2"])
        L = float(row["Length"])
        x0 = float(row["Position 1"])
        xl = xi - x0

        dofs = [2*node1, 2*node1+1, 2*node2, 2*node2+1]
        ue = u[dofs, 0]

        # Verschiebung
        w = get_Local_Element_Disp(np.array([xl]), ue, L)[0]

        # Spannungen
        sigma_top, sigma_bottom = get_Local_Element_Stress(xl, ue, L, E, z)

        disp_list.append(w)
        top_stress_list.append(sigma_top)
        bottom_stress_list.append(sigma_bottom)
        mises_stress_list.append(abs(sigma_top))

    # DataFrame erstellen
    df_results = pd.DataFrame({
        "x [mm]": x_vals,
        "Displacement [mm]": disp_list,
        "von Mises Stress [MPa]": mises_stress_list,
        "Top Stress [MPa]": top_stress_list,
        "Bottom Stress [MPa]": bottom_stress_list
    })

    csv_result = df_results.to_csv(index=True, sep=';', encoding='utf-8')

    st.subheader("Result Table")
    st.dataframe(df_results)


    st.subheader("Downloads")
    download_col1, download_col2, download_col3, download_col4, download_col5 = st.columns([1,1,1,1,1])  # Verhältnis der Breiten

    with download_col1:
        st.download_button(icon=":material/download:", label="Input Data", data=csv_data_model, file_name="beam_input.csv", mime="text/csv" )
    with download_col2:
        st.download_button(icon=":material/download:", label="K Matrix", data=csv_data_K, file_name="beam_K.csv", mime="text/csv" )
    with download_col3:
        st.download_button(icon=":material/download:", label="F Vector", data=csv_data_F, file_name="beam_F.csv", mime="text/csv" )
    with download_col4:
        st.download_button(icon=":material/download:", label="u Vector", data=csv_data_u, file_name="beam_u.csv", mime="text/csv" )
    with download_col5:
        st.download_button(icon = ":material/download:",label="Result Table", data=csv_result, file_name="beam_results.csv",mime="text/csv")


                   