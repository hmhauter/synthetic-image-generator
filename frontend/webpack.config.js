const StaticSiteGeneratorPlugin = require('static-site-generator-webpack-plugin');

module.exports = {

  entry: './src/index.tsx',
  mode: "production",

  module: {
    rules: [
      { test: /\.tsx$/, loader: 'ts-loader', exclude: '/node_modules/' },
      { test: /\.ts$/, loader: 'ts-loader', exclude: '/node_modules/' },
      {
        test: /\.scss$/i,
        use: [
          // Creates `style` nodes from JS strings
          "style-loader",
          // Translates CSS into CommonJS
          "css-loader",
          // Compiles Sass to CSS
          "sass-loader",
        ],
      },
      {
        test: /\.css$/i,
        use: [
          // Creates `style` nodes from JS strings
          "style-loader",
          // Translates CSS into CommonJS
          "css-loader",
        ],
      },
    ]
  },

  resolve: {
    extensions: ['.tsx', '.ts', '.js'],

  },

  plugins: [
    new StaticSiteGeneratorPlugin({
      paths: [
        '.src/api/',
        '.src/data/'
      ],
    //   locals: {
    //     // Properties here are merged into `locals`
    //     // passed to the exported render function
    //     greet: 'Hello'
    //   }
    })
  ],

  output: {
    filename: 'bundle.js',
    globalObject: 'this',
    path: '/dist',
    /* IMPORTANT!
     * You must compile to UMD or CommonJS
     * so it can be required in a Node context: */
    libraryTarget: 'umd'
  },


};