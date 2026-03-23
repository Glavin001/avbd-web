/**
 * AVBD WebGPU Physics Engine
 *
 * A GPU-accelerated implementation of the Augmented Vertex Block Descent (AVBD)
 * physics algorithm using WebGPU compute shaders.
 *
 * Supports both 2D and 3D rigid body simulation with a Rapier-compatible API.
 *
 * Usage:
 *   // 2D
 *   import AVBD from 'avbd-web/2d';
 *   await AVBD.init();
 *   const world = new AVBD.World({ x: 0, y: -9.81 });
 *
 *   // 3D
 *   import AVBD3D from 'avbd-web/3d';
 *   await AVBD3D.init();
 *   const world = new AVBD3D.World({ x: 0, y: -9.81, z: 0 });
 */

export { default as AVBD2D } from './2d/index.js';
export { default as AVBD3D } from './3d/index.js';

// Re-export core types
export type { Vec2, Vec3, Quat, SolverConfig } from './core/types.js';
export { RigidBodyType } from './core/types.js';

// Re-export 2D classes
export { World, RigidBody, RigidBodyDesc2D, ColliderDesc2D, JointData2D } from './2d/index.js';

// Re-export 3D classes
export { World3D, RigidBody3D, RigidBodyDesc3D, ColliderDesc3D } from './3d/index.js';

// Export core components for advanced usage
export { AVBDSolver2D } from './core/solver.js';
export { AVBDSolver3D } from './core/solver-3d.js';
export { GPUContext } from './core/gpu-context.js';
export { computeGraphColoring, validateColoring } from './core/graph-coloring.js';
