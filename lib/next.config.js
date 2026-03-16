/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config, { isServer }) => {
    // This is to solve a build error for `@tensorflow/tfjs-node` which uses `node-pre-gyp`.
    // The bundler tries to bundle a file that is not a JS module.
    // By marking it as external, we are telling Next.js to not bundle it
    // and to treat it as a regular Node.js module at runtime.
    // This is safe for server-side code.
    if (isServer) {
      config.externals.push("@mapbox/node-pre-gyp");
    }
    return config;
  },
};

module.exports = nextConfig;