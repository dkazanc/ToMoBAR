/*
CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

Copyright 2017-2020 UKRI-STFC
Copyright 2017-2020 University of Manchester

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once
#ifndef DLLEXPORT_H
#define DLLEXPORT_H

#if defined(_WIN32) || defined(__WIN32__)
#if defined(dll_EXPORTS)  // add by CMake 
#define  DLL_EXPORT __declspec(dllexport)
#define EXPIMP_TEMPLATE
#else
#define  DLL_EXPORT __declspec(dllimport)
#define EXPIMP_TEMPLATE extern
#endif 
#elif defined(linux) || defined(__linux) || defined(__APPLE__)
#define DLL_EXPORT
#ifndef __cdecl
#define __cdecl
#endif
#endif

#endif
