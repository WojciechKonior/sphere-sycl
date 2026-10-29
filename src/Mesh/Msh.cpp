#include "Mesh/Msh.hpp"

Msh::Msh(const std::string &filename)
{
    load(filename);
}

size_t Msh::get_node_count() const
{
    return nodes.size();
}

size_t Msh::get_element_count() const
{
    // Since each tetrahedron consists of exactly 4 nodes
    return 0; // elements.size();
}

const Vector3 *Msh::get_nodes_data() const
{
    return nodes.data();
}

const std::vector<int> *Msh::get_elements_data() const
{
    return nullptr; // elements.data();
}

int Msh::get_nodes_per_element(size_t element_index) const
{
    return 0; // elements.at(element_index).size(); // Tetrahedra
}

void Msh::load(const std::string &filename)
{
    if (is_msh_v4_ascii(filename))
    {
        std::cout << "[Mesh] Detected format: MSH v4 (ASCII). Loading...\n";
        load_msh_v4_ascii(filename);
        return;
    }
    else
    {
        throw std::runtime_error("[Mesh] Error: Only MSH v4 ASCII format is currently supported for .msh files.");
    }
}

bool Msh::is_msh_v4_ascii(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("[Mesh] Failed to open file for validation: " + filename);
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.find("$MeshFormat") != std::string::npos)
        {
            double version = 0.0;
            int file_type = -1; // 0 = ASCII, 1 = Binary
            int data_size = 0;

            if (file >> version >> file_type >> data_size)
            {
                if (version >= 4.0 && version < 5.0 && file_type == 0)
                {
                    return true;
                }
            }
            break;
        }
    }
    return false;
}

void Msh::load_msh_v4_ascii(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("[Mesh] Failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line))
    {
        if (line.find("$Entities") != std::string::npos)
        {
            parse_entities(file);
        }
        else if (line.find("$Nodes") != std::string::npos)
        {
            parse_nodes(file);
        }
        else if (line.find("$Elements") != std::string::npos)
        {
            parse_elements(file);
        }
    }

    file.close();
    // Obliczamy liczbę elementów 3D na podstawie rozmiarów płaskich wektorów i ich stride'ów
    size_t total_3d_elements = (elements_.tetrahedra.size() / 4) +
                               (elements_.hexahedra.size() / 8) +
                               (elements_.prisms.size() / 6) +
                               (elements_.pyramids.size() / 5);

    std::cout << "[Mesh] Successfully loaded MSH v4.1 -> Nodes: " << nodes.size()
              << ", 3D Elements Total: " << total_3d_elements
              << " (Tets: " << elements_.tetrahedra.size() / 4
              << ", Hexes: " << elements_.hexahedra.size() / 8
              << ", Prisms: " << elements_.prisms.size() / 6
              << ", Pyramids: " << elements_.pyramids.size() / 5 << ")"
              << ", Entities: " << entities.size() << "\n";
}

void Msh::parse_entities(std::ifstream &file)
{
    size_t numPoints, numCurves, numSurfaces, numVolumes;
    file >> numPoints >> numCurves >> numSurfaces >> numVolumes;

    // Points (0D)
    for (size_t i = 0; i < numPoints; ++i)
    {
        MeshEntity ent;
        ent.dim = 0;
        double x, y, z;
        size_t numPhys;
        file >> ent.tag >> x >> y >> z >> numPhys;
        ent.bbox[0] = x;
        ent.bbox[1] = y;
        ent.bbox[2] = z;
        ent.bbox[3] = x;
        ent.bbox[4] = y;
        ent.bbox[5] = z;

        ent.physicalTags.resize(numPhys);
        for (size_t p = 0; p < numPhys; ++p)
            file >> ent.physicalTags[p];
        entities.push_back(ent);
    }

    // Curves (1D)
    for (size_t i = 0; i < numCurves; ++i)
    {
        MeshEntity ent;
        ent.dim = 1;
        size_t numPhys, numBoundingPoints;
        file >> ent.tag >> ent.bbox[0] >> ent.bbox[1] >> ent.bbox[2] >> ent.bbox[3] >> ent.bbox[4] >> ent.bbox[5] >> numPhys;

        ent.physicalTags.resize(numPhys);
        for (size_t p = 0; p < numPhys; ++p)
            file >> ent.physicalTags[p];

        file >> numBoundingPoints;
        for (size_t bp = 0; bp < numBoundingPoints; ++bp)
        {
            int dummy_pt;
            file >> dummy_pt;
        }
        entities.push_back(ent);
    }

    // Surfaces (2D)
    for (size_t i = 0; i < numSurfaces; ++i)
    {
        MeshEntity ent;
        ent.dim = 2;
        size_t numPhys, numBoundingCurves;
        file >> ent.tag >> ent.bbox[0] >> ent.bbox[1] >> ent.bbox[2] >> ent.bbox[3] >> ent.bbox[4] >> ent.bbox[5] >> numPhys;

        ent.physicalTags.resize(numPhys);
        for (size_t p = 0; p < numPhys; ++p)
            file >> ent.physicalTags[p];

        file >> numBoundingCurves;
        for (size_t bc = 0; bc < numBoundingCurves; ++bc)
        {
            int dummy_crv;
            file >> dummy_crv;
        }
        entities.push_back(ent);
    }

    // Volumes (3D)
    for (size_t i = 0; i < numVolumes; ++i)
    {
        MeshEntity ent;
        ent.dim = 3;
        size_t numPhys, numBoundingSurfaces;
        file >> ent.tag >> ent.bbox[0] >> ent.bbox[1] >> ent.bbox[2] >> ent.bbox[3] >> ent.bbox[4] >> ent.bbox[5] >> numPhys;

        ent.physicalTags.resize(numPhys);
        for (size_t p = 0; p < numPhys; ++p)
            file >> ent.physicalTags[p];

        file >> numBoundingSurfaces;
        for (size_t bs = 0; bs < numBoundingSurfaces; ++bs)
        {
            int dummy_srf;
            file >> dummy_srf;
        }
        entities.push_back(ent);
    }
    std::string end_tag;
    file >> end_tag;
    if (end_tag != "$EndEntities")
    {
        throw std::runtime_error("[Mesh] Parser error: Expected $EndEntities, but found: " + end_tag);
    }
}

void Msh::parse_nodes(std::ifstream &file)
{
    size_t numEntityBlocks, numNodes, minNodeTag, maxNodeTag;
    file >> numEntityBlocks >> numNodes >> minNodeTag >> maxNodeTag;

    nodes.resize(numNodes);

    for (size_t b = 0; b < numEntityBlocks; ++b)
    {
        int entityDim, entityTag, parametric, numNodesInBlock;
        file >> entityDim >> entityTag >> parametric >> numNodesInBlock;

        std::vector<size_t> blockNodeTags(numNodesInBlock);
        for (int i = 0; i < numNodesInBlock; ++i)
        {
            file >> blockNodeTags[i];
        }

        // Liczba wartości parametrycznych dopisanych po Z dla KAŻDEGO węzła
        int num_params_per_node = (parametric == 1) ? entityDim : (parametric == 2 ? 2 * entityDim : 0);

        for (int i = 0; i < numNodesInBlock; ++i)
        {
            double x, y, z;
            file >> x >> y >> z;

            // Zjadamy wartości parametryczne u, v, w przypisane do tego konkretnego węzła
            for (int p = 0; p < num_params_per_node; ++p)
            {
                double dummy_param;
                file >> dummy_param;
            }

            size_t global_idx = blockNodeTags[i] - 1;
            if (global_idx < nodes.size())
            {
                nodes[global_idx] = {static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)};
            }
        }
    }

    std::string end_tag;
    file >> end_tag;
    if (end_tag != "$EndNodes")
    {
        throw std::runtime_error("[Mesh] Parser error: Expected $EndNodes, but found: " + end_tag);
    }
}

void Msh::parse_elements(std::ifstream &file)
{
    size_t numEntityBlocks, numElements, minElementTag, maxElementTag;
    file >> numEntityBlocks >> numElements >> minElementTag >> maxElementTag;

    for (size_t b = 0; b < numEntityBlocks; ++b)
    {
        int entityDim, entityTag, elementType;
        size_t numElementsInBlock;
        file >> entityDim >> entityTag >> elementType >> numElementsInBlock;

        // Określamy liczbę węzłów na podstawie typu elementu w Gmsh
        int nodes_per_elem = 0;
        if (elementType == 15)
            nodes_per_elem = 1; // Point (0D)
        else if (elementType == 1)
            nodes_per_elem = 2; // Line (1D)
        else if (elementType == 2)
            nodes_per_elem = 3; // Triangle (2D Boundary)
        else if (elementType == 3)
            nodes_per_elem = 4; // Quadrangle (2D Boundary)
        else if (elementType == 4)
            nodes_per_elem = 4; // Tetrahedron (3D)
        else if (elementType == 5)
            nodes_per_elem = 8; // Hexahedron (3D)
        else if (elementType == 6)
            nodes_per_elem = 6; // Prism (3D)
        else if (elementType == 7)
            nodes_per_elem = 5; // Pyramid (3D)
        else
        {
            throw std::runtime_error("[Mesh] Błąd: Nieznany elementType (" + std::to_string(elementType) + ") w pliku MSH!");
        }

        for (size_t i = 0; i < numElementsInBlock; ++i)
        {
            size_t elemTag;
            file >> elemTag;

            // Odczytujemy indeksy węzłów dla bieżącego elementu
            std::vector<int> elem_nodes(nodes_per_elem);
            for (int n = 0; n < nodes_per_elem; ++n)
            {
                size_t nodeTag;
                file >> nodeTag;
                elem_nodes[n] = static_cast<int>(nodeTag) - 1; // Konwersja na 0-based indexing
            }

            // Rozdzielamy do odpowiednich płaskich wektorów w strukturze Elements (ignorujemy punkty/linie w domenowych strukturach 3D, ale zjadamy je poprawnie ze strumienia)
            if (elementType == 4)
            {
                elements_.tetrahedra.insert(elements_.tetrahedra.end(), elem_nodes.begin(), elem_nodes.end());
            }
            else if (elementType == 5)
            {
                elements_.hexahedra.insert(elements_.hexahedra.end(), elem_nodes.begin(), elem_nodes.end());
            }
            else if (elementType == 6)
            {
                elements_.prisms.insert(elements_.prisms.end(), elem_nodes.begin(), elem_nodes.end());
            }
            else if (elementType == 7)
            {
                elements_.pyramids.insert(elements_.pyramids.end(), elem_nodes.begin(), elem_nodes.end());
            }
            else if (elementType == 2)
            {
                elements_.boundary_triangles.insert(elements_.boundary_triangles.end(), elem_nodes.begin(), elem_nodes.end());
            }
            else if (elementType == 3)
            {
                elements_.boundary_quads.insert(elements_.boundary_quads.end(), elem_nodes.begin(), elem_nodes.end());
            }
            // Typy 1 (linie) i 15 (punkty) są poprawnie odczytywane ze strumienia, dzięki czemu nie psują synchronizacji.
        }
    }

    std::string end_tag;
    file >> end_tag;
    if (end_tag != "$EndElements")
    {
        throw std::runtime_error("[Mesh] Parser error: Expected $EndElements, but found: '" + end_tag + "'");
    }
}