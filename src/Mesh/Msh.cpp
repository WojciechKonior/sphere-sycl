#include "Mesh/Msh.hpp"

Msh::Msh(const std::string &filename, sycl::queue& q) : queue_(q)
{
    load(filename);
    upload_to_device();
}

Msh::~Msh()
{
    // RAII: Bezpieczne zwolnienie alokacji USM przy niszczeniu obiektu Msh za pomocą wskaźników z widoku.
    // Używamy const_cast, ponieważ sycl::free wymaga wskaźnika nie-const, a widok trzyma wskaźniki odczytu.
    if (mesh_device_view.d_nodes)              sycl::free(const_cast<Vector3*>(mesh_device_view.d_nodes), queue_);
    if (mesh_device_view.d_hexahedra)          sycl::free(const_cast<int*>(mesh_device_view.d_hexahedra), queue_);
    if (mesh_device_view.d_tetrahedra)         sycl::free(const_cast<int*>(mesh_device_view.d_tetrahedra), queue_);
    if (mesh_device_view.d_prisms)             sycl::free(const_cast<int*>(mesh_device_view.d_prisms), queue_);
    if (mesh_device_view.d_pyramids)           sycl::free(const_cast<int*>(mesh_device_view.d_pyramids), queue_);
    if (mesh_device_view.d_boundary_quads)     sycl::free(const_cast<int*>(mesh_device_view.d_boundary_quads), queue_);
    if (mesh_device_view.d_boundary_triangles) sycl::free(const_cast<int*>(mesh_device_view.d_boundary_triangles), queue_);
}

void Msh::upload_to_device()
{
    // 1. Alokacja i kopiowanie węzłów
    if (!nodes.empty()) {
        mesh_device_view.d_nodes = sycl::malloc_device<Vector3>(nodes.size(), queue_);
        queue_.copy(nodes.data(), const_cast<Vector3*>(mesh_device_view.d_nodes), nodes.size()).wait();
        mesh_device_view.num_nodes = nodes.size();
    }

    // 2. Alokacja i kopiowanie elementów (hexa)
    if (!elements.hexahedra.empty()) {
        mesh_device_view.d_hexahedra = sycl::malloc_device<int>(elements.hexahedra.size(), queue_);
        queue_.copy(elements.hexahedra.data(), const_cast<int*>(mesh_device_view.d_hexahedra), elements.hexahedra.size()).wait();
        mesh_device_view.num_hexahedra_indices = elements.hexahedra.size();
    }

    // 3. Alokacja i kopiowanie elementów (tets)
    if (!elements.tetrahedra.empty()) {
        mesh_device_view.d_tetrahedra = sycl::malloc_device<int>(elements.tetrahedra.size(), queue_);
        queue_.copy(elements.tetrahedra.data(), const_cast<int*>(mesh_device_view.d_tetrahedra), elements.tetrahedra.size()).wait();
        mesh_device_view.num_tetrahedra_indices = elements.tetrahedra.size();
    }

    // 4. Pozostałe typy (prisms, pyramids, brzegi)...
    if (!elements.prisms.empty()) {
        mesh_device_view.d_prisms = sycl::malloc_device<int>(elements.prisms.size(), queue_);
        queue_.copy(elements.prisms.data(), const_cast<int*>(mesh_device_view.d_prisms), elements.prisms.size()).wait();
        mesh_device_view.num_prisms_indices = elements.prisms.size();
    }
    if (!elements.pyramids.empty()) {
        mesh_device_view.d_pyramids = sycl::malloc_device<int>(elements.pyramids.size(), queue_);
        queue_.copy(elements.pyramids.data(), const_cast<int*>(mesh_device_view.d_pyramids), elements.pyramids.size()).wait();
        mesh_device_view.num_pyramids_indices = elements.pyramids.size();
    }
    if (!elements.boundary_quads.empty()) {
        mesh_device_view.d_boundary_quads = sycl::malloc_device<int>(elements.boundary_quads.size(), queue_);
        queue_.copy(elements.boundary_quads.data(), const_cast<int*>(mesh_device_view.d_boundary_quads), elements.boundary_quads.size()).wait();
        mesh_device_view.num_boundary_quads_indices = elements.boundary_quads.size();
    }
    if (!elements.boundary_triangles.empty()) {
        mesh_device_view.d_boundary_triangles = sycl::malloc_device<int>(elements.boundary_triangles.size(), queue_);
        queue_.copy(elements.boundary_triangles.data(), const_cast<int*>(mesh_device_view.d_boundary_triangles), elements.boundary_triangles.size()).wait();
        mesh_device_view.num_boundary_triangles_indices = elements.boundary_triangles.size();
    }
}

MeshDeviceView Msh::get_device_view() const
{
    return mesh_device_view;
}

size_t Msh::get_node_count() const
{
    return mesh_device_view.num_nodes;
}

size_t Msh::get_element_count(ElementType type) const
{
    switch (type) {
        case ElementType::Tetrahedron:
            return mesh_device_view.num_tetrahedra_indices / 4;       // 4 indeksy na czworościan
        case ElementType::Hexahedron:
            return mesh_device_view.num_hexahedra_indices / 8;        // 8 indeksów na heksaedr
        case ElementType::Prism:
            return mesh_device_view.num_prisms_indices / 6;           // 6 indeksów na pryzmę
        case ElementType::Pyramid:
            return mesh_device_view.num_pyramids_indices / 5;         // 5 indeksów na piramidę
        case ElementType::BoundaryQuad:
            return mesh_device_view.num_boundary_quads_indices / 4;   // 4 indeksy na czworokąt brzegowy
        case ElementType::BoundaryTriangle:
            return mesh_device_view.num_boundary_triangles_indices / 3; // 3 indeksy na trójkąt brzegowy
        default:
            return 0;
    }
}

const Vector3 *Msh::get_nodes_data() const
{
    return mesh_device_view.d_nodes;
}

const int* Msh::get_elements_data(ElementType type) const
{
    switch (type) {
        case ElementType::Tetrahedron:
            return mesh_device_view.d_tetrahedra;
        case ElementType::Hexahedron:
            return mesh_device_view.d_hexahedra;
        case ElementType::Prism:
            return mesh_device_view.d_prisms;
        case ElementType::Pyramid:
            return mesh_device_view.d_pyramids;
        case ElementType::BoundaryQuad:
            return mesh_device_view.d_boundary_quads;
        case ElementType::BoundaryTriangle:
            return mesh_device_view.d_boundary_triangles;
        default:
            return nullptr;
    }
}

int Msh::get_nodes_per_element(ElementType type) const
{
    switch (type) {
        case ElementType::Tetrahedron:      return 4;
        case ElementType::Hexahedron:       return 8;
        case ElementType::Prism:            return 6;
        case ElementType::Pyramid:          return 5;
        case ElementType::BoundaryQuad:     return 4;
        case ElementType::BoundaryTriangle: return 3;
        default:                            return 0;
    }
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
    size_t total_3d_elements = (elements.tetrahedra.size() / 4) +
                               (elements.hexahedra.size() / 8) +
                               (elements.prisms.size() / 6) +
                               (elements.pyramids.size() / 5);

    std::cout << "[Mesh] Successfully loaded MSH v4.1 -> Nodes: " << nodes.size()
              << ", 3D Elements Total: " << total_3d_elements
              << " (Tets: " << elements.tetrahedra.size() / 4
              << ", Hexes: " << elements.hexahedra.size() / 8
              << ", Prisms: " << elements.prisms.size() / 6
              << ", Pyramids: " << elements.pyramids.size() / 5 << ")"
              << ", Entities: " << entities.size() << "\n";
}

void Msh::parse_entities(std::ifstream &file)
{
    constexpr std::size_t MAX_ENTITIES = 100'000'000;
    constexpr std::size_t MAX_TAGS_PER_ENTITY = 1'000'000;

    auto read_or_throw = [&file](auto& value, const char* description) {
        if (!(file >> value)) {
            throw std::runtime_error( std::string("[Mesh] Failed to read ") + description + " in $Entities");
        }
    };

    auto validate_count = [](std::size_t count, std::size_t limit, const char* description) {
        if (count > limit) {
            throw std::runtime_error(std::string("[Mesh] Excessive ") + description + " count in $Entities");
        }
    };

    // Nagłówek: liczba punktów, krzywych, powierzchni i objętości.
    std::size_t numPoints = 0, numCurves = 0,numSurfaces = 0, numVolumes = 0;

    read_or_throw(numPoints, "number of points");
    read_or_throw(numCurves, "number of curves");
    read_or_throw(numSurfaces, "number of surfaces");
    read_or_throw(numVolumes, "number of volumes");

    validate_count(numPoints, MAX_ENTITIES, "point");
    validate_count(numCurves, MAX_ENTITIES, "curve");
    validate_count(numSurfaces, MAX_ENTITIES, "surface");
    validate_count(numVolumes, MAX_ENTITIES, "volume");

    // Ochrona przed przepełnieniem sumy liczby encji.
    if (numCurves > std::numeric_limits<std::size_t>::max() - numPoints ||
        numSurfaces > std::numeric_limits<std::size_t>::max() - numPoints - numCurves ||
        numVolumes > std::numeric_limits<std::size_t>::max() - numPoints - numCurves - numSurfaces) {
        throw std::runtime_error( "[Mesh] Entity count overflow in $Entities");
    }

    const std::size_t totalEntities = numPoints + numCurves + numSurfaces + numVolumes;

    // Parsowanie jednej listy tagów fizycznych.
    auto read_physical_tags = [&](MeshEntity& ent, std::size_t numPhys) {
        validate_count(numPhys, MAX_TAGS_PER_ENTITY, "physical tag");
        ent.physicalTags.resize(numPhys);
        for (std::size_t p = 0; p < numPhys; ++p) {
            read_or_throw(ent.physicalTags[p], "physical tag");
        }
    };

    // Pominięcie listy tagów encji granicznych.
    // Odczytujemy je, mimo że nie przechowujemy ich w MeshEntity.
    auto skip_bounding_tags = [&](std::size_t count, const char* description) {
        validate_count(count, MAX_TAGS_PER_ENTITY, description);
        for (std::size_t i = 0; i < count; ++i) {
            long long tag = 0;
            read_or_throw(tag, description);
        }
    };

    // Opcjonalnie rezerwujemy miejsce na wszystkie encje.
    // Zakładamy, że entities jest std::vector<MeshEntity>.
    if (totalEntities > entities.max_size() - entities.size()) {
        throw std::runtime_error( "[Mesh] Too many entities for container capacity");
    }

    entities.reserve(entities.size() + totalEntities);

    // Points (0D)
    for (size_t i = 0; i < numPoints; ++i)
    {
        MeshEntity ent{};
        ent.dim = 0;
        double x = 0.0, y = 0.0, z = 0.0;
        size_t numPhys = 0;

        read_or_throw(ent.tag, "point tag");
        read_or_throw(x, "point X coordinate");
        read_or_throw(y, "point Y coordinate");
        read_or_throw(z, "point Z coordinate");
        read_or_throw(numPhys, "point physical tag count");
        if (!sycl::isfinite(x) || !sycl::isfinite(y) || !sycl::isfinite(z)) {
            throw std::runtime_error("[Mesh] Non-finite point coordinates in $Entities");
        }

        ent.bbox[0] = x;
        ent.bbox[1] = y;
        ent.bbox[2] = z;
        ent.bbox[3] = x;
        ent.bbox[4] = y;
        ent.bbox[5] = z;

        read_physical_tags(ent, numPhys);
        entities.push_back(std::move(ent));
    }

    // Curves (1D)
    for (size_t i = 0; i < numCurves; ++i)
    {
        MeshEntity ent{};
        ent.dim = 1;
        std::size_t numPhys = 0;
        std::size_t numBoundingCurves = 0;
        read_or_throw(ent.tag, "curve tag");
        for (std::size_t j = 0; j < 6; ++j) {
            read_or_throw(ent.bbox[j], "curve bounding box");
            if (!sycl::isfinite(ent.bbox[j])) {
                throw std::runtime_error( "[Mesh] Non-finite curve bounding box in $Entities");
            }
        }
        read_or_throw(numPhys, "curve physical tag count");
        read_physical_tags(ent, numPhys);
        read_or_throw(numBoundingCurves, "curve bounding point count");
        skip_bounding_tags(numBoundingCurves, "curve bounding point tag");
        entities.push_back(std::move(ent));
    }

    // Surfaces (2D)
    for (size_t i = 0; i < numSurfaces; ++i)
    {
        MeshEntity ent{};
        ent.dim = 2;
        std::size_t numPhys = 0;
        std::size_t numBoundingCurves = 0;
        read_or_throw(ent.tag, "surface tag");
        for (std::size_t j = 0; j < 6; ++j) {
            read_or_throw(ent.bbox[j], "surface bounding box");
            if (!sycl::isfinite(ent.bbox[j])) {
                throw std::runtime_error( "[Mesh] Non-finite surface bounding box in $Entities");
            }
        }
        read_or_throw(numPhys, "surface physical tag count");
        read_physical_tags(ent, numPhys);
        read_or_throw(numBoundingCurves, "surface bounding curve count");
        skip_bounding_tags(numBoundingCurves, "surface bounding curve tag");
        entities.push_back(std::move(ent));
    }

    // Volumes (3D)
    for (size_t i = 0; i < numVolumes; ++i)
    {
        MeshEntity ent{};
        ent.dim = 3;
        std::size_t numPhys = 0;
        std::size_t numBoundingSurfaces = 0;
        read_or_throw(ent.tag, "volume tag");
        for (std::size_t j = 0; j < 6; ++j) {
            read_or_throw(ent.bbox[j], "volume bounding box");
            if (!sycl::isfinite(ent.bbox[j])) {
                throw std::runtime_error( "[Mesh] Non-finite volume bounding box in $Entities");
            }
        }
        read_or_throw(numPhys, "volume physical tag count");
        read_physical_tags(ent, numPhys);
        read_or_throw(numBoundingSurfaces, "volume bounding surface count");
        skip_bounding_tags(numBoundingSurfaces, "volume bounding surface tag");
        entities.push_back(std::move(ent));
    }

    std::string end_tag;
    read_or_throw(end_tag, "$EndEntities");
    if (end_tag != "$EndEntities") {
        throw std::runtime_error("[Mesh] Parser error: Expected $EndEntities, but found: " + end_tag);
    }
}

void Msh::parse_nodes(std::ifstream &file)
{
    constexpr std::size_t MAX_NODES = 100'000'000;
    constexpr std::size_t MAX_ENTITY_BLOCKS = 100'000'000;

    auto read_or_throw = [&file](auto& value, const char* description) {
        if (!(file >> value)) {
            throw std::runtime_error( std::string("[Mesh] Failed to read ") + description + " in $Nodes");
        }
    };

    auto read_count = [&](const char* description, std::size_t limit) -> std::size_t {
        // Odczyt ze znakiem pozwala jawnie odrzucić wartości ujemne.
        long long value = 0;
        read_or_throw(value, description);
        if (value < 0 || static_cast<unsigned long long>(value) > static_cast<unsigned long long>(limit)) {
            throw std::runtime_error( std::string("[Mesh] Invalid or excessive ") + description + " in $Nodes");
        }
        return static_cast<std::size_t>(value);
    };

    const std::size_t numEntityBlocks = read_count("entity block count", MAX_ENTITY_BLOCKS);
    const std::size_t numNodes = read_count("node count", MAX_NODES);
    long long minNodeTag = 0;
    long long maxNodeTag = 0;
    read_or_throw(minNodeTag, "minimum node tag");
    read_or_throw(maxNodeTag, "maximum node tag");

    if (minNodeTag < 0 || maxNodeTag < 0) {
        throw std::runtime_error( "[Mesh] Negative node tag range in $Nodes");
    }

    if (numNodes == 0) {
        if (numEntityBlocks != 0 || minNodeTag != 0 || maxNodeTag != 0) {
            throw std::runtime_error( "[Mesh] Inconsistent empty $Nodes section");
        }
    }
    else {
        if (numEntityBlocks == 0 || minNodeTag <= 0 || maxNodeTag < minNodeTag) {
            throw std::runtime_error( "[Mesh] Invalid $Nodes header");
        }
    }

    // UWAGA: Poniższa implementacja zachowuje dotychczasowe mapowanie: indeks w wektorze = tag węzła - 1.
    // Jest ono poprawne tylko dla tagów 1, 2, ..., numNodes. Odrzucamy pliki z tagami nieciągłymi zamiast błędnie przypisywać współrzędne do niewłaściwych węzłów.
    if (numNodes > 0 && (minNodeTag != 1 || maxNodeTag != static_cast<long long>(numNodes))) {
        throw std::runtime_error( "[Mesh] Non-contiguous node tags are not supported by the current tag-1 indexing scheme");
    }

    nodes.resize(numNodes);
    std::vector<bool> seen(numNodes, false);
    std::size_t totalReadNodes = 0;

    for (size_t b = 0; b < numEntityBlocks; ++b)
    {
        long long entityDimRaw = 0;
        long long entityTag = 0;
        long long parametricRaw = 0;

        read_or_throw(entityDimRaw, "entity dimension");
        read_or_throw(entityTag, "entity tag");
        read_or_throw(parametricRaw, "parametric flag");

        const std::size_t numNodesInBlock = read_count("nodes in entity block", MAX_NODES);

        if (entityDimRaw < 0 || entityDimRaw > 3) { throw std::runtime_error( "[Mesh] Invalid entity dimension in $Nodes"); }
        if (entityTag <= 0) { throw std::runtime_error( "[Mesh] Invalid entity tag in $Nodes"); }
        if (parametricRaw != 0 && parametricRaw != 1) { throw std::runtime_error( "[Mesh] Invalid parametric flag in $Nodes"); }

        const int entityDim = static_cast<int>(entityDimRaw);
        const bool parametric = (parametricRaw == 1);

        if (numNodesInBlock > numNodes - totalReadNodes) { throw std::runtime_error( "[Mesh] Entity blocks contain more nodes than declared in $Nodes header"); }

        // Najpierw odczytujemy wszystkie tagi węzłów bloku,
        // a następnie współrzędne — tak definiuje to MSH 4.1.
        std::vector<std::size_t> blockNodeTags(numNodesInBlock);

        for (std::size_t i = 0; i < numNodesInBlock; ++i)
        {
            long long tag = 0;
            read_or_throw(tag, "node tag");
            if (tag <= 0 || static_cast<unsigned long long>(tag) > static_cast<unsigned long long>(numNodes)) {
                throw std::runtime_error( "[Mesh] Node tag is outside the supported contiguous range");
            }
            const std::size_t globalIdx = static_cast<std::size_t>(tag - 1);
            if (seen[globalIdx]) { throw std::runtime_error( "[Mesh] Duplicate node tag in $Nodes"); }
            seen[globalIdx] = true;
            blockNodeTags[i] = globalIdx;
        }

        // W MSH 4.1 dla węzłów parametrycznych po X, Y, Z
        // występuje entityDim dodatkowych wartości na węzeł.
        const int numParamsPerNode = parametric ? entityDim : 0;

        for (std::size_t i = 0; i < numNodesInBlock; ++i)
        {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;

            read_or_throw(x, "node X coordinate");
            read_or_throw(y, "node Y coordinate");
            read_or_throw(z, "node Z coordinate");

            if (!sycl::isfinite(x) || !sycl::isfinite(y) || !sycl::isfinite(z)) {
                throw std::runtime_error( "[Mesh] Non-finite node coordinates in $Nodes");
            }

            for (int p = 0; p < numParamsPerNode; ++p) {
                double param = 0.0;
                read_or_throw(param, "parametric node coordinate");
                if (!sycl::isfinite(param)) {
                    throw std::runtime_error( "[Mesh] Non-finite parametric coordinate in $Nodes");
                }
            }

            // Sprawdzenie, czy współrzędne mieszczą się w float.
            const float xf = static_cast<float>(x);
            const float yf = static_cast<float>(y);
            const float zf = static_cast<float>(z);

            if (!sycl::isfinite(xf) || !sycl::isfinite(yf) || !sycl::isfinite(zf)) {
                throw std::runtime_error( "[Mesh] Node coordinates exceed float range");
            }
            nodes[blockNodeTags[i]] = {xf, yf, zf};
        }
        totalReadNodes += numNodesInBlock;
    }

    if (totalReadNodes != numNodes) {
        throw std::runtime_error( "[Mesh] Total number of nodes in blocks does not match $Nodes header");
    }

    // Każdy tag z zakresu 1..numNodes powinien wystąpić dokładnie raz.
    for (std::size_t i = 0; i < numNodes; ++i) {
        if (!seen[i]) {
            throw std::runtime_error( "[Mesh] Missing node tag in $Nodes");
        }
    }

    std::string end_tag;
    read_or_throw(end_tag, "$EndNodes");
    if (end_tag != "$EndNodes") {
        throw std::runtime_error("[Mesh] Parser error: Expected $EndNodes, but found: " + end_tag);
    }
}

void Msh::parse_elements(std::ifstream& file)
{
    constexpr std::size_t MAX_ELEMENTS = 100'000'000;
    constexpr std::size_t MAX_ENTITY_BLOCKS = 100'000'000;

    auto read_or_throw = [&file](auto& value, const char* description) {
        if (!(file >> value)) {
            throw std::runtime_error( std::string("[Mesh] Failed to read ") + description + " in $Elements");
        }
    };

    auto read_count = [&](const char* description, std::size_t limit) -> std::size_t {
        long long value = 0;
        read_or_throw(value, description);
        if (value < 0 || static_cast<unsigned long long>(value) > static_cast<unsigned long long>(limit)) {
            throw std::runtime_error( std::string("[Mesh] Invalid or excessive ") + description + " in $Elements");
        }
        return static_cast<std::size_t>(value);
    };

    // Nagłówek MSH 4.1:
    // numEntityBlocks numElements minElementTag maxElementTag
    const std::size_t numEntityBlocks = read_count("entity block count", MAX_ENTITY_BLOCKS);
    const std::size_t numElements = read_count("element count", MAX_ELEMENTS);
    long long minElementTag = 0;
    long long maxElementTag = 0;
    read_or_throw(minElementTag, "minimum element tag");
    read_or_throw(maxElementTag, "maximum element tag");

    if (minElementTag < 0 || maxElementTag < 0) { throw std::runtime_error( "[Mesh] Negative element tag range in $Elements"); }

    if (numElements == 0) {
        if (numEntityBlocks != 0 || minElementTag != 0 || maxElementTag != 0) {
            throw std::runtime_error( "[Mesh] Inconsistent empty $Elements section");
        }
    }
    else if (numEntityBlocks == 0 || minElementTag <= 0 || maxElementTag < minElementTag) {
        throw std::runtime_error( "[Mesh] Invalid $Elements header");
    }

    // Obsługiwane typy elementów Gmsh:
    // typ, wymagany wymiar encji, liczba węzłów.
    struct ElementInfo
    {
        int type;
        int dimension;
        std::size_t nodeCount;
    };

    auto get_element_info = [](int type) -> ElementInfo {
        switch (type)
        {
            case 15: return {15, 0, 1}; // Point
            case 1:  return {1,  1, 2}; // Line
            case 2:  return {2,  2, 3}; // Triangle
            case 3:  return {3,  2, 4}; // Quadrangle
            case 4:  return {4,  3, 4}; // Tetrahedron
            case 5:  return {5,  3, 8}; // Hexahedron
            case 6:  return {6,  3, 6}; // Prism
            case 7:  return {7,  3, 5}; // Pyramid
            default:
                throw std::runtime_error( "[Mesh] Unsupported element type " + std::to_string(type) + " in $Elements");
        }
    };

    // Sprawdzamy, czy liczba węzłów mieści się w typie int,
    // ponieważ właśnie takim typem przechowujesz indeksy elementów.
    if (nodes.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error( "[Mesh] Too many nodes for int-based element indices");
    }

    std::size_t totalReadElements = 0;

    long long observedMinTag = std::numeric_limits<long long>::max();
    long long observedMaxTag = 0;

    for (std::size_t b = 0; b < numEntityBlocks; ++b)
    {
        long long entityDimRaw = 0;
        long long entityTag = 0;
        long long elementTypeRaw = 0;

        read_or_throw(entityDimRaw, "entity dimension");
        read_or_throw(entityTag, "entity tag");
        read_or_throw(elementTypeRaw, "element type");

        const std::size_t numElementsInBlock = read_count("elements in entity block", MAX_ELEMENTS);

        if (entityDimRaw < 0 || entityDimRaw > 3) { throw std::runtime_error( "[Mesh] Invalid entity dimension in $Elements"); }
        if (entityTag <= 0) { throw std::runtime_error( "[Mesh] Invalid entity tag in $Elements"); }
        if (elementTypeRaw < std::numeric_limits<int>::min() || elementTypeRaw > std::numeric_limits<int>::max()) {
            throw std::runtime_error( "[Mesh] Element type is outside the supported range");
        }

        const int entityDim = static_cast<int>(entityDimRaw);
        const int elementType = static_cast<int>(elementTypeRaw);

        const ElementInfo info = get_element_info(elementType);

        if (entityDim != info.dimension) {
            throw std::runtime_error( "[Mesh] Element type " + std::to_string(elementType) + " is inconsistent with entity dimension " + std::to_string(entityDim));
        }

        if (numElementsInBlock > numElements - totalReadElements) {
            throw std::runtime_error( "[Mesh] Entity blocks contain more elements than declared in $Elements header");
        }

        // Stały bufor: maksymalnie 8 węzłów dla obsługiwanych typów.
        std::array<int, 8> elemNodes{};

        for (std::size_t i = 0; i < numElementsInBlock; ++i)
        {
            long long elemTag = 0;
            read_or_throw(elemTag, "element tag");

            if (elemTag <= 0) { throw std::runtime_error( "[Mesh] Element tag must be positive"); }
            if (elemTag < minElementTag || elemTag > maxElementTag) {
                throw std::runtime_error( "[Mesh] Element tag is outside the range declared in $Elements header");
            }

            observedMinTag = std::min(observedMinTag, elemTag);
            observedMaxTag = std::max(observedMaxTag, elemTag);

            for (std::size_t n = 0; n < info.nodeCount; ++n)
            {
                long long nodeTag = 0;
                read_or_throw(nodeTag, "element node tag");

                // Przyjęte mapowanie: indeks = nodeTag - 1.
                // Wymaga ciągłych tagów węzłów 1..nodes.size().
                if (nodeTag <= 0 || static_cast<unsigned long long>(nodeTag) > static_cast<unsigned long long>(nodes.size())) {
                    throw std::runtime_error( "[Mesh] Element references an invalid node tag: " + std::to_string(nodeTag));
                }

                elemNodes[n] = static_cast<int>(nodeTag - 1);
            }

            // Zapis do odpowiedniego płaskiego wektora.
            switch (elementType)
            {
                case 4: elements.tetrahedra.insert( elements.tetrahedra.end(), elemNodes.begin(), elemNodes.begin() + 4); break;
                case 5: elements.hexahedra.insert( elements.hexahedra.end(), elemNodes.begin(), elemNodes.begin() + 8); break;
                case 6: elements.prisms.insert( elements.prisms.end(), elemNodes.begin(), elemNodes.begin() + 6); break;
                case 7: elements.pyramids.insert( elements.pyramids.end(), elemNodes.begin(), elemNodes.begin() + 5); break;
                case 2: elements.boundary_triangles.insert( elements.boundary_triangles.end(), elemNodes.begin(), elemNodes.begin() + 3); break;
                case 3: elements.boundary_quads.insert( elements.boundary_quads.end(), elemNodes.begin(), elemNodes.begin() + 4); break;
                // Typy 1 i 15 są odczytywane, ale nie przechowujemy
                // ich w strukturach elementów domenowych.
                case 1:
                case 15: break;
            }
        }
        totalReadElements += numElementsInBlock;
    }

    if (totalReadElements != numElements) { throw std::runtime_error( "[Mesh] Total number of elements in blocks does not match $Elements header"); }
    if (numElements > 0 && (observedMinTag != minElementTag || observedMaxTag != maxElementTag)) {
        throw std::runtime_error( "[Mesh] Element tag range does not match $Elements header");
    }

    std::string endTag;
    read_or_throw(endTag, "$EndElements");
    if (endTag != "$EndElements") {
        throw std::runtime_error( "[Mesh] Parser error: Expected $EndElements, but found: '" + endTag + "'");
    }
}