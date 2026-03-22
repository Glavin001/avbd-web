import { describe, it, expect } from 'vitest';
import { computeGraphColoring, validateColoring } from '../src/core/graph-coloring.js';

describe('Graph Coloring', () => {
  it('should handle empty graph', () => {
    const result = computeGraphColoring(0, []);
    expect(result).toEqual([]);
  });

  it('should color isolated bodies with single color', () => {
    const result = computeGraphColoring(5, []);
    expect(result.length).toBe(1);
    expect(result[0].bodyIndices.length).toBe(5);
  });

  it('should color a single pair with two colors', () => {
    const result = computeGraphColoring(2, [[0, 1]]);
    expect(result.length).toBe(2);
    expect(validateColoring(2, [[0, 1]], result)).toBe(true);
  });

  it('should color a triangle with 3 colors', () => {
    const pairs: [number, number][] = [[0, 1], [1, 2], [0, 2]];
    const result = computeGraphColoring(3, pairs);
    expect(result.length).toBe(3);
    expect(validateColoring(3, pairs, result)).toBe(true);
  });

  it('should color a chain with 2 colors', () => {
    // Linear chain: 0-1-2-3-4
    const pairs: [number, number][] = [[0, 1], [1, 2], [2, 3], [3, 4]];
    const result = computeGraphColoring(5, pairs);
    expect(result.length).toBe(2); // Bipartite graph
    expect(validateColoring(5, pairs, result)).toBe(true);
  });

  it('should exclude fixed bodies', () => {
    const pairs: [number, number][] = [[0, 1], [1, 2]];
    const fixed = new Set([0]); // body 0 is fixed
    const result = computeGraphColoring(3, pairs, fixed);

    // Body 0 should not appear in any color group
    for (const group of result) {
      expect(group.bodyIndices).not.toContain(0);
    }
    expect(validateColoring(3, pairs, result)).toBe(true);
  });

  it('should handle a typical stacking scenario', () => {
    // 10 boxes stacked: each touches the one below
    const pairs: [number, number][] = [];
    for (let i = 0; i < 9; i++) {
      pairs.push([i, i + 1]);
    }
    const result = computeGraphColoring(10, pairs);
    expect(result.length).toBe(2);
    expect(validateColoring(10, pairs, result)).toBe(true);

    // All bodies should be colored
    const allBodies = new Set<number>();
    for (const group of result) {
      for (const b of group.bodyIndices) {
        allBodies.add(b);
      }
    }
    expect(allBodies.size).toBe(10);
  });

  it('should handle a dense contact graph', () => {
    // 6 bodies, each touching every other (complete graph K6)
    const pairs: [number, number][] = [];
    for (let i = 0; i < 6; i++) {
      for (let j = i + 1; j < 6; j++) {
        pairs.push([i, j]);
      }
    }
    const result = computeGraphColoring(6, pairs);
    expect(result.length).toBe(6); // Complete graph needs N colors
    expect(validateColoring(6, pairs, result)).toBe(true);
  });

  it('should skip self-referencing pairs', () => {
    const result = computeGraphColoring(2, [[0, 0], [0, 1]]);
    expect(validateColoring(2, [[0, 1]], result)).toBe(true);
  });

  it('should handle duplicate pairs', () => {
    const pairs: [number, number][] = [[0, 1], [0, 1], [1, 0]];
    const result = computeGraphColoring(2, pairs);
    expect(result.length).toBe(2);
    expect(validateColoring(2, pairs, result)).toBe(true);
  });
});
