use anyhow::Result;
use std::fs::File;
use std::io::BufReader;
use std::{collections::HashMap, io::BufRead};

#[allow(non_snake_case)]
struct CsvEntry {
    ID: usize,
    GAME_NAME: String,
    BEHAVIOR: String,
    PLAY_PURCHASE: String,
}

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Node {
    id: usize,
    name: usize,
    behavior: usize,
    play_purchase: usize,
}

#[derive(Default)]
struct Graph {
    entries: Vec<Node>,
    node_data: Vec<String>,
}

impl Graph {
    fn duplicates(&self) -> HashMap<Node, usize> {
        let mut map = HashMap::new();
        for id in &self.entries {
            let count = map.entry(*id).or_insert(0);
            *count += 1;
        }
        map
    }

    fn collect<I: IntoIterator<Item = Result<CsvEntry>>>(iter: I) -> Result<Self> {
        let mut graph = Graph::default();
        let mut map = HashMap::new();

        let mut get_or_insert = |field: &str| -> usize {
            *map.entry(field.to_string()).or_insert_with(|| {
                graph.node_data.push(field.to_string());
                graph.node_data.len() - 1
            })
        };

        for entry in iter {
            let entry = entry?;
            let name = get_or_insert(&entry.GAME_NAME);
            let behavior = get_or_insert(&entry.BEHAVIOR);
            let play_purchase = get_or_insert(&entry.PLAY_PURCHASE);

            graph.entries.push(Node {
                id: entry.ID,
                name,
                behavior,
                play_purchase,
            })
        }

        Ok(graph)
    }
}

fn main() -> Result<()> {
    println!("Building graph...");

    let graph = Graph::collect(
        BufReader::new(File::open("algorithms part dataset.csv")?)
            .lines()
            .skip(1)
            .map(|line| {
                let line = line?;
                let mut iter = line.split(',');
                Ok(CsvEntry {
                    ID: iter.next().unwrap().replace('"', "").parse::<usize>()?,
                    GAME_NAME: iter.next().unwrap().to_string(),
                    BEHAVIOR: iter.next().unwrap().to_string(),
                    PLAY_PURCHASE: iter.next().unwrap().to_string(),
                })
            }),
    )?;

    println!("Finding duplicates...");

    println!(
        "Duplicates: {:#?}",
        graph
            .duplicates()
            .iter()
            .filter(|(_, dups)| **dups > 1)
            .collect::<HashMap<_, _>>()
    );
    Ok(())
}
