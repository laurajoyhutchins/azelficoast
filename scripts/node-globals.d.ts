interface AzelficoastNodeRequire {
  (id: string): any;
  main?: any;
}

declare const Buffer: any;
declare const __dirname: string;
declare const __filename: string;
declare const module: any;
declare const process: any;
declare const require: AzelficoastNodeRequire;

declare function clearImmediate(handle: any): void;
declare function setImmediate(callback: (...args: any[]) => void, ...args: any[]): any;

declare module "node:*" {
  const value: any;
  export = value;
}
