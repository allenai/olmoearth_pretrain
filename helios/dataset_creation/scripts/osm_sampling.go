package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"os"
	"time"

	"github.com/qedus/osmpbf"
)

// Size of tiles to split world into in degrees.
const TileSize = 0.001

func decodeOsm(fname string, f func(v interface{})) {
	file, err := os.Open(fname)
	if err != nil {
		panic(err)
	}
	defer file.Close()
	decoder := osmpbf.NewDecoder(file)
	decoder.SetBufferSize(osmpbf.MaxBlobSize)
	decoder.Start(64)
	for {
		if v, err := decoder.Decode(); err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		} else {
			f(v)
		}
	}
}

type Relation struct {
	Categories []string
	Exterior   []int64
}

type Way struct {
	Categories []string
	NodeIDs    []int64
}

type Node struct {
	ID         int64
	Categories []string
	Tile       [2]int
}

type Feature struct {
	Geometry   Geometry
	Properties map[string]string
}

type Geometry struct {
	Type     string
	Point    [2]int
	Polyline [][2]int
	Polygon  [][][2]int
}

func getCategories(tags map[string]string) []string {
	var categories []string
	if tags["railway"] != "" {
		if tags["tunnel"] != "yes" && tags["railway"] != "abandoned" {
			categories = append(categories, "railway")
			if tags["bridge"] != "" {
				categories = append(categories, "railway_bridge")
			}
		}
	}
	if tags["waterway"] == "river" {
		categories = append(categories, "river")
	}
	if tags["amenity"] == "fountain" {
		categories = append(categories, "fountain")
	}
	if tags["barrier"] == "toll_booth" {
		categories = append(categories, "toll_booth")
	}
	if tags["man_made"] == "chimney" {
		categories = append(categories, "chimney")
	}
	if tags["man_made"] == "crane" {
		categories = append(categories, "crane")
	}
	if tags["man_made"] == "obelisk" {
		categories = append(categories, "obelisk")
	}
	if tags["man_made"] == "observatory" {
		categories = append(categories, "observatory")
	}
	if tags["man_made"] == "satellite_dish" {
		categories = append(categories, "satellite_dish")
	}
	if tags["man_made"] == "tower" {
		categories = append(categories, "tower")
	}
	if tags["man_made"] == "works" {
		categories = append(categories, "works")
	}
	if tags["man_made"] == "flagpole" {
		categories = append(categories, "flagpole")
	}
	if tags["man_made"] == "pier" {
		categories = append(categories, "pier")
	}
	if tags["landuse"] != "" {
		landuseCategories := map[string]bool{
			"farmland":                true,
			"orchard":                 true,
			"vineyard":                true,
			"industrial":              true,
			"cemetery":                true,
			"allotments":              true,
			"reservoir":               true,
			"quarry":                  true,
			"greenhouse_horticulture": true,
			"landfill":                true,
			"winter_sports":           true,
		}
		if landuseCategories[tags["landuse"]] {
			categories = append(categories, tags["landuse"])
		}
		if tags["landuse"] == "quarry" && tags["resource"] != "" {
			validResources := map[string]bool{
				"sand":   true,
				"gravel": true,
				"clay":   true,
				"coal":   true,
				"peat":   true,
			}
			if validResources[tags["resource"]] {
				categories = append(categories, "quarry_"+tags["resource"])
			}
		}
	}
	if tags["generator:source"] == "wind" {
		categories = append(categories, "wind_turbine")
	}
	if tags["landuse"] == "aquaculture" || tags["seamark:type"] == "marine_farm" {
		categories = append(categories, "aquafarm")
	}
	if tags["man_made"] == "lighthouse" {
		categories = append(categories, "lighthouse")
	}
	if tags["water"] == "lock" {
		categories = append(categories, "lock")
	}
	if tags["waterway"] == "dam" {
		categories = append(categories, "dam")
	}
	if tags["power"] == "plant" {
		categories = append(categories, "power_plant")
		if tags["plant:source"] != "" {
			sourceCategories := map[string]bool{
				"wind":       true,
				"solar":      true,
				"biomass":    true,
				"hydro":      true,
				"coal":       true,
				"gas":        true,
				"oil":        true,
				"geothermal": true,
				"nuclear":    true,
				"waste":      true,
				"battery":    true,
			}
			if sourceCategories[tags["plant:source"]] {
				categories = append(categories, "power_plant_"+tags["plant:source"])
			}
		}
	}
	if tags["amenity"] == "fuel" {
		categories = append(categories, "gas_station")
	}
	if tags["leisure"] != "" {
		leisureCategories := map[string]bool{
			"pitch":         true,
			"swimming_pool": true,
			"park":          true,
			"garden":        true,
			"playground":    true,
			"sports_centre": true,
			"track":         true,
			"slipway":       true,
			"stadium":       true,
			"golf_course":   true,
			"marina":        true,
			"water_park":    true,
			"horse_riding":  true,
			"beach_resort":  true,
			"ice_rink":      true,
			"climbing":      true,
		}
		if leisureCategories[tags["leisure"]] {
			categories = append(categories, "leisure_"+tags["leisure"])
		}
		if tags["leisure"] == "pitch" && tags["sport"] != "" {
			validSports := map[string]string{
				"american_football": "american_football",
				"badminton":         "badminton",
				"baseball":          "baseball",
				"basketball":        "basketball",
				"cricket":           "cricket",
				"rugby":             "rugby",
				"rugby_league":      "rugby",
				"rugby_union":       "rugby",
				"soccer":            "soccer",
				"tennis":            "tennis",
				"volleyball":        "volleyball",
			}
			if validSports[tags["sport"]] != "" {
				categories = append(categories, validSports[tags["sport"]])
			}
		}
		if tags["leisure"] == "track" && tags["sport"] != "" {
			validSports := map[string]string{
				"running":      "running",
				"cycling":      "cycling",
				"horse_racing": "horse",
				"equestrian":   "horse",
			}
			if validSports[tags["sport"]] != "" {
				categories = append(categories, validSports[tags["sport"]])
			}
		}
	}
	if tags["amenity"] == "parking" {
		if tags["building"] == "parking" {
			categories = append(categories, "parking_garage")
		} else {
			categories = append(categories, "parking_lot")
		}
	}
	if tags["man_made"] == "mineshaft" {
		categories = append(categories, "mineshaft")
	}
	if tags["aeroway"] == "aerodrome" {
		categories = append(categories, "airport")
	}
	if tags["aeroway"] == "runway" {
		categories = append(categories, "airport_runway")
	}
	if tags["aeroway"] == "taxiway" {
		categories = append(categories, "airport_taxiway")
	}
	if tags["aeroway"] == "apron" {
		categories = append(categories, "airport_apron")
	}
	if tags["aeroway"] == "hangar" {
		categories = append(categories, "airport_hangar")
	}
	if tags["aeroway"] == "airstrip" {
		categories = append(categories, "airstrip")
	}
	if tags["aeroway"] == "terminal" {
		categories = append(categories, "airport_terminal")
	}
	if tags["tourism"] == "theme_park" {
		categories = append(categories, "theme_park")
	}
	if tags["man_made"] == "storage_tank" {
		categories = append(categories, "storage_tank")
	}
	if tags["man_made"] == "silo" {
		categories = append(categories, "silo")
	}
	if tags["man_made"] == "wastewater_plant" {
		categories = append(categories, "wastewater_plant")
	}
	if tags["aerialway"] == "pylon" {
		categories = append(categories, "aerialway_pylon")
	}
	if tags["man_made"] == "communications_tower" {
		categories = append(categories, "communications_tower")
	}
	if tags["aeroway"] == "helipad" {
		categories = append(categories, "helipad")
	}
	if tags["aeroway"] == "launchpad" {
		categories = append(categories, "launchpad")
	}
	if tags["man_made"] == "water_tower" {
		categories = append(categories, "water_tower")
	}
	if tags["man_made"] == "petroleum_well" {
		categories = append(categories, "petroleum_well")
	}
	if tags["power"] == "tower" {
		categories = append(categories, "power_tower")
	}
	if tags["power"] == "substation" {
		categories = append(categories, "power_substation")
	}
	if tags["shop"] == "mall" {
		categories = append(categories, "shop_mall")
	}
	if tags["shop"] == "department_store" {
		categories = append(categories, "shop_department_store")
	}
	if tags["amenity"] == "hospital" {
		categories = append(categories, "hospital")
	}
	if tags["amenity"] == "ferry_terminal" {
		categories = append(categories, "ferry_terminal")
	}
	if tags["amenity"] == "place_of_worship" {
		categories = append(categories, "place_of_worship")
	}
	if tags["natural"] == "peak" {
		categories = append(categories, "natural_peak")
	}
	if tags["man_made"] == "offshore_platform" {
		categories = append(categories, "offshore_platform")
	}
	if tags["man_made"] == "beacon" {
		categories = append(categories, "beacon")
	}
	if tags["man_made"] == "bridge" {
		categories = append(categories, "man_made_bridge")
	}
	if tags["man_made"] == "flare" {
		categories = append(categories, "flare")
	}
	if tags["man_made"] == "oil_gas_separator" {
		categories = append(categories, "oil_gas_separator")
	}
	if tags["man_made"] == "pumping_station" {
		categories = append(categories, "pumping_station")
	}
	if tags["highway"] == "trailhead" {
		categories = append(categories, "trailhead")
	}
	if tags["natural"] == "cape" {
		categories = append(categories, "natural_cape")
	}
	if tags["natural"] == "geyser" {
		categories = append(categories, "natural_geyser")
	}
	if tags["natural"] == "hot_spring" {
		categories = append(categories, "natural_hot_spring")
	}
	if tags["natural"] == "arch" {
		categories = append(categories, "natural_arch")
	}
	if tags["natural"] == "cave_entrance" {
		categories = append(categories, "natural_cave_entrance")
	}
	if tags["natural"] == "hill" {
		categories = append(categories, "natural_hill")
	}
	if tags["natural"] == "volcano" {
		categories = append(categories, "natural_volcano")
	}
	return categories
}

func main() {
	osmPath := "planet-latest.osm.pbf"
	outPath := "tiles_by_category.json"

	// First pass: decode relations.
	fmt.Println("decode relations")
	t0 := time.Now()
	count := 0
	var relations []Relation
	neededWays := make(map[int64]bool)
	decodeOsm(osmPath, func(v interface{}) {
		switch v := v.(type) {
		case *osmpbf.Relation:
			count++
			if count%1000000 == 0 {
				fmt.Printf("finished %dM relations (%v elapsed)\n", count/1000000, int(time.Now().Sub(t0).Seconds()))
			}

			categories := getCategories(v.Tags)
			if len(categories) == 0 {
				return
			}

			var relation Relation
			relation.Categories = categories
			for _, member := range v.Members {
				if member.Role == "outer" {
					relation.Exterior = append(relation.Exterior, member.ID)
					neededWays[member.ID] = true
				}
			}
			if len(relation.Exterior) == 0 {
				return
			}
			relations = append(relations, relation)
		}
	})
	fmt.Printf("got %d relations\n", len(relations))

	// Second pass: decode ways.
	fmt.Println("decode ways")
	t0 = time.Now()
	count = 0
	ways := make(map[int64]Way)
	neededNodes := make(map[int64]bool)
	decodeOsm(osmPath, func(v interface{}) {
		switch v := v.(type) {
		case *osmpbf.Way:
			count++
			if count%1000000 == 0 {
				fmt.Printf("finished %dM ways (%v elapsed)\n", count/1000000, int(time.Now().Sub(t0).Seconds()))
			}

			if len(v.NodeIDs) < 2 {
				return
			}

			categories := getCategories(v.Tags)
			if len(categories) == 0 && !neededWays[v.ID] {
				return
			}

			var way Way
			way.Categories = categories
			for _, nodeID := range v.NodeIDs {
				way.NodeIDs = append(way.NodeIDs, nodeID)
				neededNodes[nodeID] = true
			}
			ways[v.ID] = way
		}
	})
	fmt.Printf("got %d ways\n", len(ways))

	// Third pass: decode nodes, just record their mercator column/row.
	fmt.Println("decode nodes")
	t0 = time.Now()
	nodes := make(map[int64]Node)
	count = 0
	decodeOsm(osmPath, func(v interface{}) {
		switch v := v.(type) {
		case *osmpbf.Node:
			count++
			if count%10000000 == 0 {
				fmt.Printf("finished %dM nodes (%v elapsed)\n", count/1000000, int(time.Now().Sub(t0).Seconds()))
			}

			categories := getCategories(v.Tags)
			if len(categories) == 0 && !neededNodes[v.ID] {
				return
			}

			tile := [2]int{int(v.Lon / TileSize), int(v.Lat / TileSize)}
			nodes[v.ID] = Node{
				ID:         v.ID,
				Categories: categories,
				Tile:       tile,
			}
		}
	})
	fmt.Printf("got %d nodes\n", len(nodes))

	fmt.Println("preparing tile categories")
	tileCategories := make(map[string]map[[2]int]bool)

	for _, node := range nodes {
		categories := node.Categories
		if len(categories) == 0 {
			continue
		}
		tile := node.Tile

		for _, category := range categories {
			if tileCategories[category] == nil {
				tileCategories[category] = make(map[[2]int]bool)
			}
			tileCategories[category][tile] = true
		}
	}

	getWayNodes := func(nodeIDs []int64) []Node {
		var curNodes []Node
		for _, nodeID := range nodeIDs {
			node, ok := nodes[nodeID]
			if !ok {
				continue
			}
			curNodes = append(curNodes, node)
		}
		return curNodes
	}

	for _, way := range ways {
		if len(way.Categories) == 0 {
			continue
		}
		nodes := getWayNodes(way.NodeIDs)
		if len(nodes) == 0 {
			continue
		}
		randIdx := rand.Intn(len(nodes))
		tile := nodes[randIdx].Tile
		for _, category := range way.Categories {
			if tileCategories[category] == nil {
				tileCategories[category] = make(map[[2]int]bool)
			}
			tileCategories[category][tile] = true
		}
	}

	for _, relation := range relations {
		if len(relation.Categories) == 0 {
			continue
		}

		var allNodes []Node
		for _, wayID := range relation.Exterior {
			way, ok := ways[wayID]
			if !ok {
				continue
			}
			curNodes := getWayNodes(way.NodeIDs)
			allNodes = append(allNodes, curNodes...)
		}
		if len(allNodes) == 0 {
			continue
		}
		randIdx := rand.Intn(len(allNodes))
		tile := allNodes[randIdx].Tile
		for _, category := range relation.Categories {
			if tileCategories[category] == nil {
				tileCategories[category] = make(map[[2]int]bool)
			}
			tileCategories[category][tile] = true
		}
	}

	fmt.Println("writing tile categories")
	tilesByCategory := make(map[string][][2]int)
	for category, tileSet := range tileCategories {
		for tile := range tileSet {
			tilesByCategory[category] = append(tilesByCategory[category], tile)
		}
	}
	bytes, err := json.Marshal(tilesByCategory)
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(outPath, bytes, 0644); err != nil {
		panic(err)
	}
}
