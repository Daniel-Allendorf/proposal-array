use std::time::Instant;

use dynamic_weighted_index::DynamicWeightedIndex;
use pcg_rand::Pcg64;
use rand::distributions::Distribution;
use rand::Rng;
use rand::SeedableRng;


fn benchmark_insert(nl: usize, nu: usize, f: f64, samples: usize) {
    let mut rng = Pcg64::from_entropy();

    let mut dyn_index: DynamicWeightedIndex<f64> = DynamicWeightedIndex::new(nu);
    for i in 0..nl {
        let w = rng.gen_range(0.0..(nl as f64));
        dyn_index.set_weight(i, w);
    }

    let mut n = nl;
    while n < nu {
        let start = Instant::now();
        for _i in 0..samples {
            let _s = dyn_index.sample(&mut rng).unwrap();
        }
        let runtime = start.elapsed();
        println!("WeightedIndex Insert [n: {}] Time elapsed: {}ms", n, runtime.as_millis());

        let nf = ((n as f64) * f) as usize;
	if nf >= nu { break; }
        while n < nf {
            let w = rng.gen_range(0.0..(nl as f64));
            dyn_index.set_weight(n, w);
            n = n + 1;
        }
    }
}

fn benchmark_erase(nl: usize, nu: usize, f: f64, samples: usize) {
    let mut rng = Pcg64::from_entropy();

    let mut dyn_index: DynamicWeightedIndex<f64> = DynamicWeightedIndex::new(nu);
    for i in 0..nu {
        let w = rng.gen_range(0.0..(nl as f64));
        dyn_index.set_weight(i, w);
    }

    let mut n = nu;
    while n > ((nl as f64) / f) as usize {
        let start = Instant::now();
        for _i in 0..samples {
            let _s = dyn_index.sample(&mut rng).unwrap();
        }
        let runtime = start.elapsed();
        println!("WeightedIndex Erase [n: {}] Time elapsed: {}ms", n, runtime.as_millis());

        let nf = ((n as f64) / f) as usize;
        while n > nf {
            dyn_index.remove_weight(n - 1);
            n = n - 1;
        }
    }
}

fn benchmark_random_increase(n: usize, g: usize, samples: usize) {
    let mut rng = Pcg64::from_entropy();

    let mut weights = Vec::new();

    let mut dyn_index: DynamicWeightedIndex<f64> = DynamicWeightedIndex::new(n);
    for i in 0..n {
        let w = rng.gen_range(0.0..(n as f64));
        weights.push(w);
        dyn_index.set_weight(i, w);
    }

    let steps = 100 * n;
    let substeps = steps / g;
    let mut t = 0;
    while t < steps {
        let start = Instant::now();
        for _i in 0..samples {
            let _s = dyn_index.sample(&mut rng).unwrap();
        }
        let runtime = start.elapsed();
        println!("WeightedIndex RandomIncrease [n: {}] Time elapsed: {}ms", t, runtime.as_millis());

        for _i in 0..substeps {
            let w = rng.gen_range(0.0..(n as f64));
            let u = rng.gen_range(0..n);
            weights[u] += w;
            dyn_index.set_weight(u, weights[u]);
        }
        t += substeps;
    }
}

fn benchmark_polya_urn(n: usize, g: usize, samples: usize) {
    let mut rng = Pcg64::from_entropy();

    let mut weights = Vec::new();

    let mut dyn_index: DynamicWeightedIndex<f64> = DynamicWeightedIndex::new(n);
    for i in 0..n {
        let w = rng.gen_range(0.0..(n as f64));
        weights.push(w);
        dyn_index.set_weight(i, w);
    }

    let steps = 100 * n;
    let substeps = steps / g;
    let mut t = 0;
    while t < steps {
        let start = Instant::now();
        for _i in 0..samples {
            let _s = dyn_index.sample(&mut rng).unwrap();
        }
        let runtime = start.elapsed();
        println!("WeightedIndex PolyaUrn [n: {}] Time elapsed: {}ms", t, runtime.as_millis());

        for _i in 0..substeps {
            let w = rng.gen_range(0.0..(n as f64));
            let u = dyn_index.sample(&mut rng).unwrap();
            weights[u] += w;
            dyn_index.set_weight(u, weights[u]);
        }
        t += substeps;
    }
}

fn benchmark_single_increase(n: usize, g: usize, samples: usize) {
    let mut rng = Pcg64::from_entropy();

    let mut weights = Vec::new();

    let mut dyn_index: DynamicWeightedIndex<f64> = DynamicWeightedIndex::new(n);
    for i in 0..n {
        let w = rng.gen_range(0.0..(n as f64));
        weights.push(w);
        dyn_index.set_weight(i, w);
    }

    let steps = 100 * n;
    let substeps = steps / g;
    let mut t = 0;
    while t < steps {
        let start = Instant::now();
        for _i in 0..samples {
            let _s = dyn_index.sample(&mut rng).unwrap();
        }
        let runtime = start.elapsed();
        println!("WeightedIndex SingleIncrease [n: {}] Time elapsed: {}ms", t, runtime.as_millis());

        for _i in 0..substeps {
            let w = rng.gen_range(0.0..(n as f64));
            let u = 0;
            weights[u] += w;
            dyn_index.set_weight(u, weights[u]);
        }
        t += substeps;
    }
}

fn main() {
    let samples = 1000000;
    let repeats = 10;

    let nl = 1<<16;
    let nu = 1<<26;
    let f = (2 as f64).sqrt().sqrt();

    let n = 10000000;
    let g = 100;

    for _r in 0..repeats {
        benchmark_insert(nl, nu, f, samples);
        benchmark_erase(nl, nu, f, samples);
        benchmark_random_increase(n, g, samples);
        benchmark_polya_urn(n, g, samples);
        benchmark_single_increase(n, g, samples);
    }
}