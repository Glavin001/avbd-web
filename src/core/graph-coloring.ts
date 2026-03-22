/**
 * Graph coloring for parallel dispatch grouping.
 * Bodies sharing a constraint cannot be in the same color group
 * (they write to each other's data during the primal update).
 *
 * Uses greedy graph coloring — simple and effective for the typical
 * rigid body contact graphs (usually 5-20 colors needed).
 */

import type { ColorGroup } from './types.js';

/**
 * Build an adjacency list from constraint pairs.
 * @param numBodies Total number of bodies
 * @param constraintPairs Array of [bodyA, bodyB] index pairs
 * @returns Adjacency list: adjacency[i] = set of body indices adjacent to body i
 */
function buildAdjacency(numBodies: number, constraintPairs: [number, number][]): Set<number>[] {
  const adjacency: Set<number>[] = Array.from({ length: numBodies }, () => new Set());
  for (const [a, b] of constraintPairs) {
    if (a >= 0 && b >= 0 && a < numBodies && b < numBodies && a !== b) {
      adjacency[a].add(b);
      adjacency[b].add(a);
    }
  }
  return adjacency;
}

/**
 * Greedy graph coloring.
 * Assigns each body a color such that no two adjacent bodies share a color.
 *
 * @param numBodies Total number of bodies
 * @param constraintPairs Pairs of body indices that share constraints
 * @param fixedBodyIndices Indices of fixed/static bodies (excluded from coloring)
 * @returns Array of ColorGroup, each containing body indices of the same color
 */
export function computeGraphColoring(
  numBodies: number,
  constraintPairs: [number, number][],
  fixedBodyIndices: Set<number> = new Set(),
): ColorGroup[] {
  if (numBodies === 0) return [];

  const adjacency = buildAdjacency(numBodies, constraintPairs);
  const colors = new Int32Array(numBodies).fill(-1);

  // Order vertices by degree (highest first) for better coloring
  const order = Array.from({ length: numBodies }, (_, i) => i);
  order.sort((a, b) => adjacency[b].size - adjacency[a].size);

  let maxColor = -1;

  for (const body of order) {
    // Skip fixed bodies
    if (fixedBodyIndices.has(body)) continue;

    // Find colors used by neighbors
    const usedColors = new Set<number>();
    for (const neighbor of adjacency[body]) {
      if (colors[neighbor] >= 0) {
        usedColors.add(colors[neighbor]);
      }
    }

    // Assign the smallest available color
    let color = 0;
    while (usedColors.has(color)) color++;
    colors[body] = color;
    if (color > maxColor) maxColor = color;
  }

  // Group bodies by color
  const groups: Map<number, number[]> = new Map();
  for (let i = 0; i < numBodies; i++) {
    if (colors[i] < 0) continue; // skip uncolored (fixed bodies)
    const c = colors[i];
    if (!groups.has(c)) groups.set(c, []);
    groups.get(c)!.push(i);
  }

  const result: ColorGroup[] = [];
  for (const [color, bodyIndices] of groups) {
    result.push({ color, bodyIndices });
  }

  // Sort by color index for determinism
  result.sort((a, b) => a.color - b.color);
  return result;
}

/**
 * Validate that a coloring is correct: no two adjacent bodies share a color.
 */
export function validateColoring(
  numBodies: number,
  constraintPairs: [number, number][],
  colorGroups: ColorGroup[],
): boolean {
  // Build body → color map
  const bodyColor = new Map<number, number>();
  for (const group of colorGroups) {
    for (const body of group.bodyIndices) {
      bodyColor.set(body, group.color);
    }
  }

  // Check all constraint pairs
  for (const [a, b] of constraintPairs) {
    const ca = bodyColor.get(a);
    const cb = bodyColor.get(b);
    if (ca !== undefined && cb !== undefined && ca === cb) {
      return false;
    }
  }
  return true;
}
