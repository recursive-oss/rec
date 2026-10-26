package main

import "embed"

//go:embed clients/opencode/recursive-logo/pixels.json
var recursiveLogo []byte

//go:embed clients/opencode/pins.json
var openCodePins []byte

//go:embed clients/opencode/recursive-logo
var openCodeLogo embed.FS

//go:embed clients/pi/package.json
var piPackage []byte

//go:embed clients/omp/package.json
var ompPackage []byte

//go:embed clients/omp/package-lock.json
var ompLock []byte

//go:embed clients/pi/package-lock.json
var piLock []byte

//go:embed clients/pi/flint-pi.ts clients/pi/pi-transport.mjs clients/pi/pi-logo.mjs clients/opencode/instructions.md
var clientAssets embed.FS
