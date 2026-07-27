//
// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2014-2024, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.
//
// NVIDIA_COPYRIGHT_END
//

#ifndef __NVRTC_H__
#define __NVRTC_H__

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>


/*************************************************************************//**
 *
 * \defgroup error Error Handling
 *
 * NVRTC defines the following enumeration type and function for API call
 * error handling.
 *
 ****************************************************************************/


/**
 * \ingroup error
 * \brief   The enumerated type nvrtcResult defines API call result codes.
 *          NVRTC API functions return nvrtcResult to indicate the call
 *          result.
 */
typedef enum {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_INVALID_INPUT = 3,
  NVRTC_ERROR_INVALID_PROGRAM = 4,
  NVRTC_ERROR_INVALID_OPTION = 5,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  NVRTC_ERROR_INTERNAL_ERROR = 11,
  NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
  NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13,
  NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14,
  NVRTC_ERROR_PCH_CREATE = 15,
  NVRTC_ERROR_CANCELLED = 16
} nvrtcResult;


/**
 * \ingroup error
 * \brief   nvrtcGetErrorString is a helper function that returns a string
 *          describing the given nvrtcResult code, e.g., NVRTC_SUCCESS to
 *          \c "NVRTC_SUCCESS".
 *          For unrecognized enumeration values, it returns
 *          \c "NVRTC_ERROR unknown".
 *
 * \param   [in] result CUDA Runtime Compilation API result code.
 * \return  Message string for the given #nvrtcResult code.
 */
const char *nvrtcGetErrorString(nvrtcResult result);


/*************************************************************************//**
 *
 * \defgroup query General Information Query
 *
 * NVRTC defines the following function for general information query.
 *
 ****************************************************************************/


/**
 * \ingroup query
 * \brief   nvrtcVersion sets the output parameters \p major and \p minor
 *          with the CUDA Runtime Compilation version number.
 *
 * \param   [out] major CUDA Runtime Compilation major version number.
 * \param   [out] minor CUDA Runtime Compilation minor version number.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *
 */
nvrtcResult nvrtcVersion(int *major, int *minor);


/**
 * \ingroup query
 * \brief   nvrtcGetNumSupportedArchs sets the output parameter \p numArchs 
 *          with the number of architectures supported by NVRTC. This can 
 *          then be used to pass an array to ::nvrtcGetSupportedArchs to
 *          get the supported architectures.
 *
 * \param   [out] numArchs number of supported architectures.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *
 * see    ::nvrtcGetSupportedArchs
 */
nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs);


/**
 * \ingroup query
 * \brief   nvrtcGetSupportedArchs populates the array passed via the output parameter 
 *          \p supportedArchs with the architectures supported by NVRTC. The array is
 *          sorted in the ascending order. The size of the array to be passed can be
 *          determined using ::nvrtcGetNumSupportedArchs.
 *
 * \param   [out] supportedArchs sorted array of supported architectures.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *
 * see    ::nvrtcGetNumSupportedArchs
 */
nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs);


/*************************************************************************//**
 *
 * \defgroup compilation Compilation
 *
 * NVRTC defines the following type and functions for actual compilation.
 *
 ****************************************************************************/


/**
 * \ingroup compilation
 * \brief   nvrtcProgram is the unit of compilation, and an opaque handle for
 *          a program.
 *
 * To compile a CUDA program string, an instance of nvrtcProgram must be
 * created first with ::nvrtcCreateProgram, then compiled with
 * ::nvrtcCompileProgram.
 */
typedef struct _nvrtcProgram *nvrtcProgram;


/**
 * \ingroup compilation
 * \brief   nvrtcCreateProgram creates an instance of nvrtcProgram with the
 *          given input parameters, and sets the output parameter \p prog with
 *          it.
 *
 * \param   [out] prog         CUDA Runtime Compilation program.
 * \param   [in]  src          CUDA program source.
 * \param   [in]  name         CUDA program name.\n
 *                             \p name can be \c NULL; \c "default_program" is
 *                             used when \p name is \c NULL or "".
 * \param   [in]  numHeaders   Number of headers used.\n
 *                             \p numHeaders must be greater than or equal to 0.
 * \param   [in]  headers      Sources of the headers.\n
 *                             \p headers can be \c NULL when \p numHeaders is
 *                             0.
 * \param   [in]  includeNames Name of each header by which they can be
 *                             included in the CUDA program source.\n
 *                             \p includeNames can be \c NULL when \p numHeaders
 *                             is 0. These headers must be included with the exact
 *                             names specified here.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_PROGRAM_CREATION_FAILURE \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcDestroyProgram
 */
nvrtcResult nvrtcCreateProgram(nvrtcProgram *prog,
                               const char *src,
                               const char *name,
                               int numHeaders,
                               const char * const *headers,
                               const char * const *includeNames);


/**
 * \ingroup compilation
 * \brief   nvrtcDestroyProgram destroys the given program.
 *
 * \param    [in] prog CUDA Runtime Compilation program.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcCreateProgram
 */
nvrtcResult nvrtcDestroyProgram(nvrtcProgram *prog);


/**
 * \ingroup compilation
 * \brief   nvrtcCompileProgram compiles the given program.
 *
 * \param   [in] prog       CUDA Runtime Compilation program.
 * \param   [in] numOptions Number of compiler options passed.
 * \param   [in] options    Compiler options in the form of C string array.\n
 *                          \p options can be \c NULL when \p numOptions is 0.
 *
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_OUT_OF_MEMORY \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_OPTION \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_COMPILATION \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_BUILTIN_OPERATION_FAILURE \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_TIME_FILE_WRITE_FAILED \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_CANCELLED \endlink
 *
 * It supports compile options listed in \ref options.
 */
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog,
                                int numOptions, const char * const *options);


/**
 * \ingroup compilation
 * \brief   nvrtcGetPTXSize sets the value of \p ptxSizeRet with the size of the PTX
 *          generated by the previous compilation of \p prog (including the
 *          trailing \c NULL).
 *
 * \param   [in]  prog       CUDA Runtime Compilation program.
 * \param   [out] ptxSizeRet Size of the generated PTX (including the trailing
 *                           \c NULL).
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetPTX
 */
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t *ptxSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetPTX stores the PTX generated by the previous compilation
 *          of \p prog in the memory pointed by \p ptx.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] ptx  Compiled result.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetPTXSize
 */
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char *ptx);


/**
 * \ingroup compilation
 * \brief   nvrtcGetCUBINSize sets the value of \p cubinSizeRet with the size of the cubin
 *          generated by the previous compilation of \p prog. The value of
 *          cubinSizeRet is set to 0 if the value specified to \c -arch is a
 *          virtual architecture instead of an actual architecture.
 *
 * \param   [in]  prog       CUDA Runtime Compilation program.
 * \param   [out] cubinSizeRet Size of the generated cubin.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetCUBIN
 */
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t *cubinSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetCUBIN stores the cubin generated by the previous compilation
 *          of \p prog in the memory pointed by \p cubin. No cubin is available
 *          if the value specified to \c -arch is a virtual architecture instead
 *          of an actual architecture.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] cubin  Compiled and assembled result.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetCUBINSize
 */
nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char *cubin);


#if defined(_WIN32)
# define __DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __DEPRECATED__(msg) __attribute__((deprecated))
#elif (defined(__GNUC__))
# define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
#else
# define __DEPRECATED__(msg)
#endif

/**
 * \ingroup compilation
 * \brief   
 * DEPRECATION NOTICE: This function will be removed in a future release. Please use
 * nvrtcGetLTOIRSize (and nvrtcGetLTOIR) instead.
 */
__DEPRECATED__("This function will be removed in a future release. Please use nvrtcGetLTOIRSize instead")
nvrtcResult nvrtcGetNVVMSize(nvrtcProgram prog, size_t *nvvmSizeRet);

/**
 * \ingroup compilation
 * \brief   
 * DEPRECATION NOTICE: This function will be removed in a future release. Please use
 * nvrtcGetLTOIR (and nvrtcGetLTOIRSize) instead.
 */
__DEPRECATED__("This function will be removed in a future release. Please use nvrtcGetLTOIR instead")
nvrtcResult nvrtcGetNVVM(nvrtcProgram prog, char *nvvm);

#undef __DEPRECATED__

/**
 * \ingroup compilation
 * \brief   nvrtcGetLTOIRSize sets the value of \p LTOIRSizeRet with the size of the LTO IR
 *          generated by the previous compilation of \p prog. The value of
 *          LTOIRSizeRet is set to 0 if the program was not compiled with 
 *          \c -dlto.
 *
 * \param   [in]  prog       CUDA Runtime Compilation program.
 * \param   [out] LTOIRSizeRet Size of the generated LTO IR.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetLTOIR
 */
nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t *LTOIRSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetLTOIR stores the LTO IR generated by the previous compilation
 *          of \p prog in the memory pointed by \p LTOIR. No LTO IR is available
 *          if the program was compiled without \c -dlto.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] LTOIR Compiled result.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetLTOIRSize
 */
nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char *LTOIR);


/**
 * \ingroup compilation
 * \brief   nvrtcGetOptiXIRSize sets the value of \p optixirSizeRet with the size of the OptiX IR
 *          generated by the previous compilation of \p prog. The value of
 *          nvrtcGetOptiXIRSize is set to 0 if the program was compiled with 
 *          options incompatible with OptiX IR generation.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] optixirSizeRet Size of the generated LTO IR.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetOptiXIR
 */
nvrtcResult nvrtcGetOptiXIRSize(nvrtcProgram prog, size_t *optixirSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetOptiXIR stores the OptiX IR generated by the previous compilation
 *          of \p prog in the memory pointed by \p optixir. No OptiX IR is available
 *          if the program was compiled with options incompatible with OptiX IR generation.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] optixir Optix IR Compiled result.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetOptiXIRSize
 */
nvrtcResult nvrtcGetOptiXIR(nvrtcProgram prog, char *optixir);

/**
 * \ingroup compilation
 * \brief   nvrtcGetProgramLogSize sets \p logSizeRet with the size of the
 *          log generated by the previous compilation of \p prog (including the
 *          trailing \c NULL).
 *
 * Note that compilation log may be generated with warnings and informative
 * messages, even when the compilation of \p prog succeeds.
 *
 * \param   [in]  prog       CUDA Runtime Compilation program.
 * \param   [out] logSizeRet Size of the compilation log
 *                           (including the trailing \c NULL).
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetProgramLog
 */
nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t *logSizeRet);


/**
 * \ingroup compilation
 * \brief   nvrtcGetProgramLog stores the log generated by the previous
 *          compilation of \p prog in the memory pointed by \p log.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [out] log  Compilation log.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *
 * \see     ::nvrtcGetProgramLogSize
 */
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char *log);


/**
 * \ingroup compilation
 * \brief   nvrtcAddNameExpression notes the given name expression
 *          denoting the address of a __global__ function 
 *          or __device__/__constant__ variable.
 *
 * The identical name expression string must be provided on a subsequent
 * call to nvrtcGetLoweredName to extract the lowered name.
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [in] name_expression constant expression denoting the address of
 *               a __global__ function or __device__/__constant__ variable.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION \endlink
 *
 * \see     ::nvrtcGetLoweredName
 */
nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog,
                                   const char * const name_expression);

/**
 * \ingroup compilation
 * \brief   nvrtcGetLoweredName extracts the lowered (mangled) name
 *          for a __global__ function or __device__/__constant__ variable,
 *          and updates *lowered_name to point to it. The memory containing
 *          the name is released when the NVRTC program is destroyed by 
 *          nvrtcDestroyProgram.
 *          The identical name expression must have been previously
 *          provided to nvrtcAddNameExpression.
 *
 * \param   [in]  prog CUDA Runtime Compilation program.
 * \param   [in] name_expression constant expression denoting the address of 
 *               a __global__ function or __device__/__constant__ variable.
 * \param   [out] lowered_name initialized by the function to point to a
 *               C string containing the lowered (mangled)
 *               name corresponding to the provided name expression.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID \endlink
 *
 * \see     ::nvrtcAddNameExpression
 */
nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                const char *const name_expression,
                                const char** lowered_name);


/*************************************************************************//**
 *
 * \defgroup precompiled_header Precompiled header (PCH) (CUDA 12.8+)
 *
 * NVRTC defines the following function related to PCH. Also see PCH related
 * flags passed to nvrtcCompileProgram.
 ****************************************************************************/


/**
 * \ingroup precompiled_header
 * \brief   retrieve the current size of the PCH Heap.
 *
 * \param   [out] ret pointer to location where the size of the PCH Heap
 *                 will be stored
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 *
 */
nvrtcResult nvrtcGetPCHHeapSize(size_t* ret);

/**
 * \ingroup precompiled_header
 * \brief   set the size of the PCH Heap.
 *
 * \param   [in] size requested size of the PCH Heap, in bytes
 * 
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 * 
 * The requested size may be rounded up to a platform dependent
 * alignment (e.g. page size). If the PCH Heap has already been allocated,
 * the heap memory will be freed and a new PCH Heap will be allocated. 
 */
nvrtcResult nvrtcSetPCHHeapSize(size_t size);

/**
 * \ingroup precompiled_header
 * \brief   returns the PCH creation status.
 *
 * \param   [in] prog CUDA Runtime Compilation program.
 *
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_PCH_CREATE \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 * 
 * NVRTC_SUCCESS indicates that the PCH was successfully created.
 * NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED indicates that no PCH creation 
 * was attempted, either because PCH functionality was not requested during
 * the preceding nvrtcCompileProgram call, or automatic PCH processing was
 * requested, and compiler chose not to create a PCH file.
 * NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED indicates that a PCH file could 
 * potentially have been created, but the compiler ran out space in the PCH
 * heap. In this scenario, the nvrtcGetPCHHeapSizeRequired() can be used to 
 * query the required heap size, the heap can be reallocated for this size with
 * nvrtcSetPCHHeapSize() and PCH creation may be reattempted again invoking
 * nvrtcCompileProgram() with a new NVRTC program instance.
 * NVRTC_ERROR_PCH_CREATE indicates that an error condition prevented the 
 * PCH file from being created. 
 */
nvrtcResult nvrtcGetPCHCreateStatus(nvrtcProgram prog);

/**
 * \ingroup precompiled_header
 * \brief   retrieve the required size of the PCH heap required to compile
 *          the given program.
 *
 * \param   [in] prog CUDA Runtime Compilation program.
 * \param   [out] size pointer to location where the required size of the PCH Heap
 *                will be stored
 *
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 * The size retrieved using this function is only valid if nvrtcGetPCHCreateStatus()
 * returned NVRTC_SUCCESS or NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED
 */
nvrtcResult nvrtcGetPCHHeapSizeRequired(nvrtcProgram prog, size_t* size);

/**
 * \ingroup compilation
 * \brief   nvrtcSetFlowCallback registers a callback function that the compiler 
 *          will invoke at different points during a call to nvrtcCompileProgram, 
 *          and the callback function can decide whether to cancel compilation by
 *          returning specific values.
 * 
 * The callback function must satisfy the following constraints:
 * 
 * (1) Its signature should be:
 *     @code
 *     int callback(void* param1, void* param2);
 *     @endcode
 *     When invoking the callback, the compiler will always pass \p payload to 
 *     param1 so that the callback may make decisions based on \p payload . It'll
 *     always pass NULL to param2 for now which is reserved for future extensions.
 * 
 * (2) It must return 1 to cancel compilation or 0 to continue. 
 *     Other return values are reserved for future use. 
 * 
 * (3) It must return consistent values. Once it returns 1 at one point, it must
 *     return 1 in all following invocations during the current nvrtcCompileProgram
 *     call in progress.
 * 
 * (4) It must be thread-safe.
 * 
 * (5) It must not invoke any nvrtc/libnvvm/ptx APIs.
 *  
 * \param   [in] prog CUDA Runtime Compilation program.
 * \param   [in] callback the callback that issues cancellation signal.
 * \param   [in] payload to be passed as a parameter when invoking the callback.
 * \return
 *   - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_PROGRAM \endlink
 *   - \link #nvrtcResult NVRTC_ERROR_INVALID_INPUT \endlink
 */
nvrtcResult nvrtcSetFlowCallback(nvrtcProgram prog, int (*callback)(void*, void*), void *payload);

/**
 * \defgroup options Supported Compile Options
 *
 * NVRTC supports the compile options below.
 * Option names with two preceding dashs (\c --) are long option names and
 * option names with one preceding dash (\c -) are short option names.
 * Short option names can be used instead of long option names.
 * When a compile option takes an argument, an assignment operator (\c =)
 * is used to separate the compile option argument from the compile option
 * name, e.g., \c "--gpu-architecture=compute_60".
 * Alternatively, the compile option name and the argument can be specified in
 * separate strings without an assignment operator, .e.g,
 * \c "--gpu-architecture" \c "compute_60".
 * Single-character short option names, such as \c -D, \c -U, and \c -I, do
 * not require an assignment operator, and the compile option name and the
 * argument can be present in the same string with or without spaces between
 * them.
 * For instance, \c "-D=<def>", \c "-D<def>", and \c "-D <def>" are all
 * supported.
 *
 * The valid compiler options are:
 *
 *   - Compilation targets
 *     - \c --gpu-architecture=\<arch\> (\c -arch)
 * 
 *       Specify the name of the class of GPU architectures for which the
 *       input must be compiled.\n
 *       - Valid <c>\<arch\></c>s:
 *         - \c compute_50
 *         - \c compute_52
 *         - \c compute_53
 *         - \c compute_60
 *         - \c compute_61
 *         - \c compute_62
 *         - \c compute_70
 *         - \c compute_72
 *         - \c compute_75
 *         - \c compute_80
 *         - \c compute_86
 *         - \c compute_87
 *         - \c compute_89
 *         - \c compute_90
 *         - \c compute_90a
 *         - \c compute_100
 *         - \c compute_100f
 *         - \c compute_100a
 *         - \c compute_101
 *         - \c compute_101f
 *         - \c compute_101a
 *         - \c compute_103
 *         - \c compute_103f
 *         - \c compute_103a
 *         - \c compute_120
 *         - \c compute_120f
 *         - \c compute_120a
 *         - \c compute_121
 *         - \c compute_121f
 *         - \c compute_121a
 *         - \c sm_50
 *         - \c sm_52
 *         - \c sm_53
 *         - \c sm_60
 *         - \c sm_61
 *         - \c sm_62
 *         - \c sm_70
 *         - \c sm_72
 *         - \c sm_75
 *         - \c sm_80
 *         - \c sm_86
 *         - \c sm_87
 *         - \c sm_89
 *         - \c sm_90
 *         - \c sm_90a
 *         - \c sm_100
 *         - \c sm_100f
 *         - \c sm_100a
 *         - \c sm_101
 *         - \c sm_101f
 *         - \c sm_101a
 *         - \c sm_103
 *         - \c sm_103f
 *         - \c sm_103a
 *         - \c sm_120
 *         - \c sm_120f
 *         - \c sm_120a
 *         - \c sm_121
 *         - \c sm_121f
 *         - \c sm_121a
 *       - Default: \c compute_52
 *   - Separate compilation / whole-program compilation
 *     - \c --device-c (\c -dc)
 *
 *       Generate relocatable code that can be linked with other relocatable
 *       device code.  It is equivalent to \c --relocatable-device-code=true.
 *     - \c --device-w (\c -dw)
 * 
 *       Generate non-relocatable code.  It is equivalent to \c --relocatable-device-code=false.
 *     - \c --relocatable-device-code={true|false} (\c -rdc)
 * 
 *       Enable (disable) the generation of relocatable device code.
 *       - Default: \c false
 *     - \c --extensible-whole-program (\c -ewp)
 * 
 *       Do extensible whole program compilation of device code.
 *       - Default: \c false
 *   - Debugging support
 *     - \c --device-debug (\c -G)
 * 
 *       Generate debug information. If \c --dopt is not specified, then turns off all optimizations.
 *     - \c --generate-line-info (\c -lineinfo)
 * 
 *       Generate line-number information.
 *   - Code generation
 *     - \c --dopt \c on (\c -dopt)
 *
 *     - \c --dopt=on
 * 
 *       Enable device code optimization. When specified along with \c -G, enables
 *       limited debug information generation for optimized device code (currently,
 *       only line number information).  When \c -G is not specified, \c -dopt=on is implicit.
 * 
 *     - \c --Ofast-compile={0|min|mid|max} (\c -Ofc)
 *       
 *       Specify the fast-compile level for device code, which controls the
 *       tradeoff between compilation speed and runtime performance by disabling
 *       certain optimizations at varying levels.
 *       - Levels:
 *         - \c max: Focus only on the fastest compilation speed, disabling
 *           many optimizations.
 *         - \c mid: Balance compile time and runtime, disabling expensive
 *           optimizations.
 *         - \c min: More minimal impact on both compile time and runtime,
 *           minimizing some expensive optimizations.
 *         - \c 0: Disable fast-compile.
 *       - The option is disabled by default.
 * 
 *     - \c --ptxas-options \<options\> (\c -Xptxas)
 * 
 *     - \c --ptxas-options=\<options\>
 * 
 *       Specify options directly to ptxas, the PTX optimizing assembler.
 *     - \c --maxrregcount=\<N\> (\c -maxrregcount)
 * 
 *       Specify the maximum amount of registers that GPU functions can use.
 *       Until a function-specific limit, a higher value will generally
 *       increase the performance of individual GPU threads that execute this
 *       function.  However, because thread registers are allocated from a
 *       global register pool on each GPU, a higher value of this option will
 *       also reduce the maximum thread block size, thereby reducing the amount
 *       of thread parallelism.  Hence, a good maxrregcount value is the result
 *       of a trade-off.  If this option is not specified, then no maximum is
 *       assumed.  Value less than the minimum registers required by ABI will
 *       be bumped up by the compiler to ABI minimum limit.
 * 
 *     - \c --ftz={true|false} (\c -ftz)
 *  
 *       When performing single-precision floating-point operations, flush
 *       denormal values to zero or preserve denormal values.
 * 
 *       \c --use_fast_math implies \c --ftz=true.
 *       - Default: \c false
 * 
 *     - \c --prec-sqrt={true|false} (\c -prec-sqrt)
 * 
 *       For single-precision floating-point square root, use IEEE
 *       round-to-nearest mode or use a faster approximation.
 *       \c --use_fast_math implies \c --prec-sqrt=false.
 *       - Default: \c true
 * 
 *     - \c --prec-div={true|false} (\c -prec-div)
 *       For single-precision floating-point division and reciprocals, use IEEE
 *       round-to-nearest mode or use a faster approximation. 
 *       \c --use_fast_math implies \c --prec-div=false.
 *       - Default: \c true
 *  
 *     - \c --fmad={true|false} (\c -fmad)
 * 
 *       Enables (disables) the contraction of floating-point multiplies and
 *       adds/subtracts into floating-point multiply-add operations (FMAD,
 *       FFMA, or DFMA).  \c --use_fast_math implies \c --fmad=true.
 *       - Default: \c true
 * 
 *     - \c --use_fast_math (\c -use_fast_math)
 * 
 *       Make use of fast math operations.
 *       \c --use_fast_math implies \c --ftz=true \c --prec-div=false
 *       \c --prec-sqrt=false \c --fmad=true.
 * 
 *     - \c --extra-device-vectorization (\c -extra-device-vectorization)
 * 
 *       Enables more aggressive device code vectorization in the NVVM optimizer.
 * 
 *     - \c --modify-stack-limit={true|false} (\c -modify-stack-limit)
 * 
 *       On Linux, during compilation, use \c setrlimit() to increase stack size 
 *       to maximum allowed. The limit is reset to the previous value at the
 *       end of compilation.
 *       Note: \c setrlimit() changes the value for the entire process.
 *       - Default: \c true
 * 
 *     - \c --dlink-time-opt (\c -dlto)
 * 
 *       Generate intermediate code for later link-time optimization.
 *       It implies \c -rdc=true. 
 *       Note: when this option is used the \c nvrtcGetLTOIR API should be used, 
 *       as PTX or Cubin will not be generated.
 * 
 *     - \c --gen-opt-lto (\c -gen-opt-lto)
 * 
 *       Run the optimizer passes before generating the LTO IR.
 *
 *     - \c --optix-ir (\c -optix-ir)
 * 
 *       Generate OptiX IR. The Optix IR is only intended for consumption by OptiX
 *       through appropriate APIs. This feature is not supported with 
 *       link-time-optimization (\c -dlto).
 * 
 *       Note: when this option is used the nvrtcGetOptiX API should be used, 
 *       as PTX or Cubin will not be generated.
 * 
 *     - \c --jump-table-density=[0-101] (\c -jtd)
 * 
 *       Specify the case density percentage in switch statements, and use it as
 *       a minimal threshold to determine whether jump table(brx.idx instruction)
 *       will be used to implement a switch statement. Default value is 101. The
 *       percentage ranges from 0 to 101 inclusively.
 * 
 *     - \c --device-stack-protector={true|false} (\c -device-stack-protector)
 * 
 *       Enable (disable) the generation of stack canaries in device code.
 * 
 *       - Default: \c false
 * 
 *     - \c --no-cache (\c -no-cache)
 *      
 *       Disable the use of cache for both ptx and cubin code generation.
 * 
 *     - \c --frandom-seed (\c -frandom-seed)
 *      
 *      The user specified random seed will be used to replace random numbers used 
 *      in generating symbol names and variable names. The option can be used to 
 *      generate deterministicly identical ptx and object files. If the input value
 *      is a valid number (decimal, octal, or hex), it will be used directly as the
 *      random seed. Otherwise, the CRC value of the passed string will be used instead.
 * 
 *   - Preprocessing
 *     - \c --define-macro=\<def\> (\c -D)
 * 
 *       \c \<def\> can be either \c \<name\> or \c \<name=definitions\>.
 *       - \c \<name\>
 * 
 *         Predefine \c \<name\> as a macro with definition \c 1.
 *       - \c \<name\>=\<definition\>
 * 
 *         The contents of \c \<definition\> are tokenized and preprocessed
 *         as if they appeared during translation phase three in a \c \#define
 *         directive.  In particular, the definition will be truncated by
 *         embedded new line characters.
 * 
 *     - \c --undefine-macro=\<def\> (\c -U)
 * 
 *       Cancel any previous definition of \c \<def\>.
 * 
 *     - \c --include-path=\<dir\> (\c -I)
 * 
 *       Add the directory \c \<dir\> to the list of directories to be
 *       searched for headers.  These paths are searched after the list of
 *       headers given to ::nvrtcCreateProgram.
 * 
 *     - \c --pre-include=\<header\> (\c -include)
 * 
 *       Preinclude \c \<header\> during preprocessing.
 * 
 *     - \c --no-source-include (\c -no-source-include)
 * 
 *       The preprocessor by default adds the directory of each input sources
 *       to the include path. This option disables this feature and only
 *       considers the path specified explicitly.
 *
 *   - Language Dialect
 *     - \c --std={c++03|c++11|c++14|c++17|c++20} (\c -std)
 * 
 *       Set language dialect to C++03, C++11, C++14, C++17 or C++20
 *       - Default: \c c++17
 * 
 *     - \c --builtin-move-forward={true|false} (\c -builtin-move-forward)
 * 
 *       Provide builtin definitions of \c std::move and \c std::forward,
 *       when C++11 or later language dialect is selected.
 *       - Default: \c true
 * 
 *     - \c --builtin-initializer-list={true|false}
 *       (\c -builtin-initializer-list)
 * 
 *       Provide builtin definitions of \c std::initializer_list class and
 *       member functions when C++11 or later language dialect is selected.
 *       - Default: \c true
 * 
 *   - Precompiled header support (CUDA 12.8+)
 *     - \c --pch (\c -pch)
 * 
 *       Enable automatic PCH processing.
 * 
 *     - \c --create-pch=<file-name> (\c -create-pch)
 * 
 *       Create a PCH file.
 * 
 *     - \c --use-pch=<file-name> (\c -use-pch)
 * 
 *       Use the specified PCH file.
 * 
 *     - \c --pch-dir=<directory-name> (\c -pch-dir)
 * 
 *       When using automatic PCH (\c -pch), look for and create PCH files in the 
 *       specified directory. When using explicit PCH (\c -create-pch or \c -use-pch),
 *       the directory name is prefixed before the specified file name, unless
 *       the file name is an absolute path name. 
 * 
 *     - \c --pch-verbose={true|false} (\c -pch-verbose)
 * 
 *       In automatic PCH mode, for each PCH file that could not be used in current
 *       compilation, print the reason in the compilation log.
 *       - Default: \c true 
 * 
 *     - \c --pch-messages={true|false} (\c -pch-messages)
 * 
 *       Print a message in the compilation log, if a PCH file was created or used
 *       in the current compilation.
 *       - Default: \c true 
 * 
 *     - \c --instantiate-templates-in-pch={true|false} (\c -instantiate-templates-in-pch)
 * 
 *       Enable or disable instantiatiation of templates before PCH creation. Instantiating
 *       templates may increase the size of the PCH file, while reducing the compilation 
 *       cost when using the PCH file (since some template instantiations can be skipped).
 *       - Default: \c true 
 * 
 *   - Misc.
 *     - \c --disable-warnings (\c -w)
 * 
 *       Inhibit all warning messages.
 *
 *     - \c --restrict (\c -restrict)
 * 
 *       Programmer assertion that all kernel pointer parameters are restrict
 *       pointers.
 * 
 *     - \c --device-as-default-execution-space
 *       (\c -default-device)
 * 
 *       Treat entities with no execution space annotation as \c __device__
 *       entities.
 * 
 *     - \c --device-int128 (\c -device-int128)
 * 
 *       Allow the \c __int128 type in device code. Also causes the macro \c __CUDACC_RTC_INT128__
 *       to be defined.
 * 
 *     - \c --device-float128 (\c -device-float128)
 * 
 *       Allow the \c __float128 and \c _Float128 types in device code. Also
 *       causes the macro \c D__CUDACC_RTC_FLOAT128__ to be defined.
 * 
 *     - \c --optimization-info=\<kind\> (\c -opt-info)
 * 
 *       Provide optimization reports for the specified kind of optimization.
 *       The following kind tags are supported:
 *         - \c inline : emit a remark when a function is inlined.
 *
 *     - \c --display-error-number (\c -err-no)
 * 
 *       Display diagnostic number for warning messages. (Default)
 * 
 *     - \c --no-display-error-number (\c -no-err-no)
 * 
 *       Disables the display of a diagnostic number for warning messages.
 * 
 *     - \c --diag-error=<error-number>,... (\c -diag-error)
 * 
 *       Emit error for specified diagnostic message number(s). Message numbers can be separated by comma.
 * 
 *     - \c --diag-suppress=<error-number>,... (\c -diag-suppress)
 * 
 *       Suppress specified diagnostic message number(s). Message numbers can be separated by comma.
 * 
 *     - \c --diag-warn=<error-number>,... (\c -diag-warn)
 * 
 *       Emit warning for specified diagnostic message number(s). Message numbers can be separated by comma.
 * 
 *     - \c --brief-diagnostics={true|false}  (\c -brief-diag)
 * 
 *       This option disables or enables showing source line and column info 
 *       in a diagnostic.
 *       The \c --brief-diagnostics=true will not show the source line and column info.
 *       - Default: \c false
 * 
 *     - \c --time=<file-name> (\c -time)
 * 
 *        Generate a comma separated value table with the time taken by each compilation
 *        phase, and append it at the end of the file given as the option argument.
 *       If the file does not exist, the column headings are generated in the first row
 *       of the table. If the file name is '-', the timing data is written to the compilation log.
 * 
 *     - \c --split-compile=<number-of-threads> (\c -split-compile=<number-of-threads>)
 * 
 *       Perform compiler optimizations in parallel.
 *       Split compilation attempts to reduce compile time by enabling the compiler to run certain
 *       optimization passes concurrently. This option accepts a numerical value that specifies the
 *       maximum number of threads the compiler can use. One can also allow the compiler to use the maximum
 *       threads available on the system by setting \c --split-compile=0.
 *       Setting \c --split-compile=1 will cause this option to be ignored.
 * 
 *     - \c --fdevice-syntax-only (\c -fdevice-syntax-only)
 * 
 *       Ends device compilation after front-end syntax checking. This option does not generate valid
 *       device code.
 *
 *     - \c --minimal  (\c -minimal)
 * 
 *        Omit certain language features to reduce compile time for small programs. 
 *        In particular, the following are omitted: 
 *            - Texture and surface functions and associated types, e.g., \c cudaTextureObject_t.
 *            - CUDA Runtime Functions that are provided by the cudadevrt device code library, 
 *              typically named with prefix "cuda", e.g., \c cudaMalloc.
 *            - Kernel launch from device code.
 *            - Types and macros associated with CUDA Runtime and Driver APIs, 
 *              provided by \c cuda/tools/cudart/driver_types.h, typically named with prefix "cuda", e.g., \c cudaError_t.
 * 
 *     - \c --device-stack-protector (\c -device-stack-protector)
 * 
 *      Enable stack canaries in device code.
 *      Stack canaries make it more difficult to exploit certain types of memory safety bugs involving
 *      stack-local variables. The compiler uses heuristics to assess the risk of such a bug in each function.
 *      Only those functions which are deemed high-risk make use of a stack canary.
 * 
 *     - \c --fdevice-time-trace=<file-name> (\c -fdevice-time-trace=<file-name>)
 *      Enables the time profiler, outputting a JSON file based on given <file-name>. Results can be analyzed on
 *      chrome://tracing for a flamegraph visualization.
 * 
 */

#ifdef __cplusplus
}
#endif /* __cplusplus */


/* The utility function 'nvrtcGetTypeName' is not available by default. Define
   the macro 'NVRTC_GET_TYPE_NAME' to a non-zero value to make it available.
*/
   
#if NVRTC_GET_TYPE_NAME || __DOXYGEN_ONLY__

#if NVRTC_USE_CXXABI || __clang__ || __GNUC__ || __DOXYGEN_ONLY__
#include <cxxabi.h>
#include <cstdlib>

#elif defined(_WIN32)
#include <Windows.h>
#include <DbgHelp.h>
#endif /* NVRTC_USE_CXXABI || __clang__ || __GNUC__ */


#include <string>
#include <typeinfo>

template <typename T> struct __nvrtcGetTypeName_helper_t { };

/*************************************************************************//**
 *
 * \defgroup hosthelper Host Helper
 *
 * NVRTC defines the following functions for easier interaction with host code.
 *
 ****************************************************************************/

/**
 * \ingroup hosthelper
 * \brief   nvrtcGetTypeName stores the source level name of a type in the given 
 *          std::string location. 
 *
 * This function is only provided when the macro NVRTC_GET_TYPE_NAME is
 * defined with a non-zero value. It uses abi::__cxa_demangle or UnDecorateSymbolName
 * function calls to extract the type name, when using gcc/clang or cl.exe compilers,
 * respectively. If the name extraction fails, it will return NVRTC_INTERNAL_ERROR,
 * otherwise *result is initialized with the extracted name.
 * 
 * Windows-specific notes:
 * - nvrtcGetTypeName() is not multi-thread safe because it calls UnDecorateSymbolName(), 
 *   which is not multi-thread safe.
 * - The returned string may contain Microsoft-specific keywords such as __ptr64 and __cdecl.
 *
 * \param   [in] tinfo: reference to object of type std::type_info for a given type.
 * \param   [in] result: pointer to std::string in which to store the type name.
 * \return
 *  - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *  - \link #nvrtcResult NVRTC_ERROR_INTERNAL_ERROR \endlink
 *
 */
inline nvrtcResult nvrtcGetTypeName(const std::type_info &tinfo, std::string *result)
{
#if USE_CXXABI || __clang__ || __GNUC__
  const char *name = tinfo.name();
  int status;
  char *undecorated_name = abi::__cxa_demangle(name, 0, 0, &status);
  if (status == 0) {
    *result = undecorated_name;
    free(undecorated_name);
    return NVRTC_SUCCESS;
  }
#elif defined(_WIN32)
  const char *name = tinfo.raw_name();
  if (!name || *name != '.') {
    return NVRTC_ERROR_INTERNAL_ERROR;
  }
  char undecorated_name[4096];
  //name+1 skips over the '.' prefix
  if(UnDecorateSymbolName(name+1, undecorated_name,
                          sizeof(undecorated_name) / sizeof(*undecorated_name),
                           //note: doesn't seem to work correctly without UNDNAME_NO_ARGUMENTS.
                           UNDNAME_NO_ARGUMENTS | UNDNAME_NAME_ONLY ) ) {
    *result = undecorated_name;
    return NVRTC_SUCCESS;
  }
#endif  /* USE_CXXABI || __clang__ || __GNUC__ */

  return NVRTC_ERROR_INTERNAL_ERROR;
}

/**
 * \ingroup hosthelper
 * \brief   nvrtcGetTypeName stores the source level name of the template type argument
 *          T in the given std::string location.
 *
 * This function is only provided when the macro NVRTC_GET_TYPE_NAME is
 * defined with a non-zero value. It uses abi::__cxa_demangle or UnDecorateSymbolName
 * function calls to extract the type name, when using gcc/clang or cl.exe compilers,
 * respectively. If the name extraction fails, it will return NVRTC_INTERNAL_ERROR,
 * otherwise *result is initialized with the extracted name.
 * 
 * Windows-specific notes:
 * - nvrtcGetTypeName() is not multi-thread safe because it calls UnDecorateSymbolName(), 
 *   which is not multi-thread safe.
 * - The returned string may contain Microsoft-specific keywords such as __ptr64 and __cdecl.
 *
 * \param   [in] result: pointer to std::string in which to store the type name.
 * \return
 *  - \link #nvrtcResult NVRTC_SUCCESS \endlink
 *  - \link #nvrtcResult NVRTC_ERROR_INTERNAL_ERROR \endlink
 *
 */
 
template <typename T>
nvrtcResult nvrtcGetTypeName(std::string *result)
{
  nvrtcResult res = nvrtcGetTypeName(typeid(__nvrtcGetTypeName_helper_t<T>), 
                                     result);
  if (res != NVRTC_SUCCESS) 
    return res;

  std::string repr = *result;
  std::size_t idx = repr.find("__nvrtcGetTypeName_helper_t");
  idx = (idx != std::string::npos) ? repr.find("<", idx) : idx;
  std::size_t last_idx = repr.find_last_of('>');
  if (idx == std::string::npos || last_idx == std::string::npos) {
    return NVRTC_ERROR_INTERNAL_ERROR;
  }
  ++idx;
  *result = repr.substr(idx, last_idx - idx);
  return NVRTC_SUCCESS;
}

#endif  /* NVRTC_GET_TYPE_NAME */

#endif /* __NVRTC_H__ */
