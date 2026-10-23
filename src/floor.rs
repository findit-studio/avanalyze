//! The crate's system floor: the oldest systems every entry point is
//! built on, and the one check each constructor makes before anything
//! else.
//!
//! The floor is written once, in [`floor!`]. The public
//! [`SYSTEM_FLOOR`] text, the `available!` check and the version the
//! test seam compares against each expand from that list, so the three
//! cannot disagree.
//!
//! # What the floor covers
//!
//! A refusal below the floor needs a process to refuse in. The loader
//! starts a process only where it can load every library the binary links
//! strongly and bind every symbol it imports strongly, before `main` and
//! so before any floor check, and a binary may be launched on any system
//! its deployment target allows, far below the floor. So everything this
//! crate links or imports strongly, as built with objc2's default features,
//! exists on every system from macOS 10.13, iOS 11 and tvOS 11 on, the
//! first row's two frameworks being the newest: from there up to the floor
//! a process starts, and every constructor refuses. The census, every
//! framework the crate links, every Vision class, global, constant and
//! selector it references, and every Foundation and CoreFoundation class,
//! selector and symbol it reaches (the Foundation tables below the Vision
//! ones), with the release that brought it (`API_AVAILABLE` in the SDK's
//! headers; visionOS follows iOS), against the floor of macOS 15, iOS 18,
//! tvOS 18 and visionOS 2:
//!
//! | what | how it is reached | introduced (macOS, iOS, tvOS) |
//! |---|---|---|
//! | Vision, and CoreML, which `objc2-vision`'s dependencies link beside it | linked strongly, by those crates' own `#[link]`, which no library can weaken | 10.13, 11, 11 (visionOS 1): a process that links this crate starts only where both exist. rustc's own defaults deploy lower on three targets (x86_64 macOS 10.12, iOS 10, tvOS 10); a binary deployed there that must start on the systems the two predate links both weakly, `-C link-arg=-Wl,-weak_framework,Vision` and the same for `CoreML`, which ld64 and lld both honour beside the strong link |
//! | every other framework and library linked (Foundation, CoreFoundation, CoreGraphics, CoreVideo, CoreImage, CoreMedia, AVFoundation, ImageIO, Metal, libobjc, libc++, libSystem, libiconv), and every symbol imported from them | linked and imported strongly. Foundation is a strong dependency twice over: `objc2-foundation` links it by its own `#[link]`, and `build.rs` links it by name (`cargo:rustc-link-lib=framework=Foundation`), because the exception barrier's `catch (NSException *)` binds `_OBJC_EHTYPE_$_NSException`, which the link resolves through Foundation (CoreFoundation's text stub exports it, Foundation re-exports CoreFoundation), and a barrier must not lose that link if a dependency's directive ever moves | each older than the oldest systems rustc supports: macOS 10.12, iOS 10, tvOS 10, visionOS 1 |
//! | every framework the crate reaches only by name, where the linker drops it | Vision loaded by `dlopen` from its system path once the floor is met, before any class is looked up, and kept loaded (`ffi::vision`). A binary linked with `-dead_strip_dylibs` drops the load command of a framework it imports no symbol from: Vision always, for this crate imports none. Foundation and CoreFoundation are not in that set and do not wait on Vision's load: both are strong dependencies (the row above), and the Foundation classes the crate looks up by name (the tables below) are found in the Foundation the binary itself links. The dead-strip integration binary (`tests/dead_strip.rs`) shows it: `otool -L` lists Foundation and CoreFoundation and not Vision. No CoreML class or symbol is reached at all | — |
//! | every class below | by name at run time (objc2's `objc_getClass`, its default), after the floor check | 10.13 to 15.0, each at or below the floor |
//! | every class below, where a consumer enables `objc2/unstable-static-class` | objc2 then emits each class as a static `OBJC_CLASS_$_…` reference, a strong link-time import: by its own documentation, "OS version < Class introduced version: Dynamic linker error on program startup" (objc2 0.6.5, `src/macros/mod.rs` 33–47, 509–513) | the class's own release: such a binary does not launch below it, so [`BelowFloor`](crate::AnalyzeErrorKind::BelowFloor) cannot answer there. The consumer's trade-off for opting into an unstable feature that changes linking, which this crate does not enable and no library can undo |
//! | every selector below | by name (objc2's `sel_registerName` at run time; the native shim's selector reference, which dyld registers at load and which imports no symbol), sent to its class or an instance of it, after the floor check | 10.13 to 15.0, each at or below the floor |
//! | the four `…JointsGroupNameAll` keys | by name, after the floor check (`ffi::vision_string`); never imported | body and hand 11.0, 14, 14; animal and 3-D body 14.0, 17, 17 |
//! | the revision constants and enumeration values below | integers fixed at compile time | no symbol |
//!
//! The classes: every Vision class the product code names, and every
//! superclass on its chain (`objc2-vision`'s `extern_class!` definitions),
//! with the release each header gives it:
//!
//! | class | its superclasses | introduced |
//! |---|---|---|
//! | `VNRequest` | `NSObject` | 10.13, 11, 11 |
//! | `VNImageBasedRequest` | `VNRequest`, `NSObject` | 10.13, 11, 11 |
//! | `VNSequenceRequestHandler` | `NSObject` | 10.13, 11, 11 |
//! | `VNObservation` | `NSObject` | 10.13, 11, 11 |
//! | `VNDetectedObjectObservation` | `VNObservation`, `NSObject` | 10.13, 11, 11 |
//! | `VNRectangleObservation` | `VNDetectedObjectObservation`, `VNObservation`, `NSObject` | 10.13, 11, 11 |
//! | `VNFaceObservation` | `VNDetectedObjectObservation`, `VNObservation`, `NSObject` | 10.13, 11, 11 |
//! | `VNBarcodeObservation` | `VNRectangleObservation`, `VNDetectedObjectObservation`, `VNObservation`, `NSObject` | 10.13, 11, 11 |
//! | `VNClassificationObservation` | `VNObservation`, `NSObject` | 10.13, 11, 11 |
//! | `VNHorizonObservation` | `VNObservation`, `NSObject` | 10.13, 11, 11 |
//! | `VNPixelBufferObservation` | `VNObservation`, `NSObject` | 10.13, 11, 11 |
//! | `VNFaceLandmarks` | `NSObject` | 10.13, 11, 11 |
//! | `VNFaceLandmarks2D` | `VNFaceLandmarks`, `NSObject` | 10.13, 11, 11 |
//! | `VNFaceLandmarkRegion` | `NSObject` | 10.13, 11, 11 |
//! | `VNFaceLandmarkRegion2D` | `VNFaceLandmarkRegion`, `NSObject` | 10.13, 11, 11 |
//! | `VNDetectBarcodesRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.13, 11, 11 |
//! | `VNDetectFaceLandmarksRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.13, 11, 11 |
//! | `VNDetectFaceRectanglesRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.13, 11, 11 |
//! | `VNDetectHorizonRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.13, 11, 11 |
//! | `VNRecognizedObjectObservation` | `VNDetectedObjectObservation`, `VNObservation`, `NSObject` | 10.14, 12, 12 |
//! | `VNRecognizedTextObservation` | `VNRectangleObservation`, `VNDetectedObjectObservation`, `VNObservation`, `NSObject` | 10.15, 13, 13 |
//! | `VNRecognizedText` | `NSObject` | 10.15, 13, 13 |
//! | `VNSaliencyImageObservation` | `VNPixelBufferObservation`, `VNObservation`, `NSObject` | 10.15, 13, 13 |
//! | `VNClassifyImageRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.15, 13, 13 |
//! | `VNDetectFaceCaptureQualityRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.15, 13, 13 |
//! | `VNDetectHumanRectanglesRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.15, 13, 13 |
//! | `VNGenerateAttentionBasedSaliencyImageRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.15, 13, 13 |
//! | `VNGenerateObjectnessBasedSaliencyImageRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.15, 13, 13 |
//! | `VNRecognizeAnimalsRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.15, 13, 13 |
//! | `VNRecognizeTextRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 10.15, 13, 13 |
//! | `VNPoint` | `NSObject` | 11.0, 14, 14 |
//! | `VNDetectedPoint` | `VNPoint`, `NSObject` | 11.0, 14, 14 |
//! | `VNRecognizedPoint` | `VNDetectedPoint`, `VNPoint`, `NSObject` | 11.0, 14, 14 |
//! | `VNRecognizedPointsObservation` | `VNObservation`, `NSObject` | 11.0, 14, 14 |
//! | `VNHumanBodyPoseObservation` | `VNRecognizedPointsObservation`, `VNObservation`, `NSObject` | 11.0, 14, 14 |
//! | `VNHumanHandPoseObservation` | `VNRecognizedPointsObservation`, `VNObservation`, `NSObject` | 11.0, 14, 14 |
//! | `VNRecognizedPoints3DObservation` | `VNObservation`, `NSObject` | 11.0, 14, 14 |
//! | `VNStatefulRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 11.0, 14, 14 |
//! | `VNDetectHumanBodyPoseRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 11.0, 14, 14 |
//! | `VNDetectHumanHandPoseRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 11.0, 14, 14 |
//! | `VNHumanObservation` | `VNDetectedObjectObservation`, `VNObservation`, `NSObject` | 12.0, 15, 15 |
//! | `VNDetectDocumentSegmentationRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 12.0, 15, 15 |
//! | `VNGeneratePersonSegmentationRequest` | `VNStatefulRequest`, `VNImageBasedRequest`, `VNRequest`, `NSObject` | 12.0, 15, 15 |
//! | `VNPoint3D` | `NSObject` | 14.0, 17, 17 |
//! | `VNRecognizedPoint3D` | `VNPoint3D`, `NSObject` | 14.0, 17, 17 |
//! | `VNHumanBodyRecognizedPoint3D` | `VNRecognizedPoint3D`, `VNPoint3D`, `NSObject` | 14.0, 17, 17 |
//! | `VNAnimalBodyPoseObservation` | `VNRecognizedPointsObservation`, `VNObservation`, `NSObject` | 14.0, 17, 17 |
//! | `VNHumanBodyPose3DObservation` | `VNRecognizedPoints3DObservation`, `VNObservation`, `NSObject` | 14.0, 17, 17 |
//! | `VNInstanceMaskObservation` | `VNObservation`, `NSObject` | 14.0, 17, 17 |
//! | `VNDetectAnimalBodyPoseRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 14.0, 17, 17 |
//! | `VNDetectHumanBodyPose3DRequest` | `VNStatefulRequest`, `VNImageBasedRequest`, `VNRequest`, `NSObject` | 14.0, 17, 17 |
//! | `VNGeneratePersonInstanceMaskRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 14.0, 17, 17 |
//! | `VNImageAestheticsScoresObservation` | `VNObservation`, `NSObject` | 15.0, 18, 18 (visionOS 2): the floor |
//! | `VNCalculateImageAestheticsScoresRequest` | `VNImageBasedRequest`, `VNRequest`, `NSObject` | 15.0, 18, 18 (visionOS 2): the floor |
//! | the protocol `VNFaceObservationAccepting`, adopted by `VNDetectFaceCaptureQualityRequest` and `VNDetectFaceLandmarksRequest` | | 10.13, 11, 11 |
//!
//! The revision constants and enumeration values, compiled in as integers,
//! each with the release its header gives it:
//!
//! | constant or value | introduced |
//! |---|---|
//! | `VNDetectHorizonRequestRevision1` | 10.14, 12, 12 |
//! | `VNDetectFaceLandmarksRequestRevision3` | 10.15, 13, 13 |
//! | `VNRequestTextRecognitionLevel` `Accurate`, `Fast` | no annotation of its own: the text request's, 10.15, 13, 13 |
//! | `VNDetectHumanBodyPoseRequestRevision1`, `VNDetectHumanHandPoseRequestRevision1` | 11.0, 14, 14 |
//! | `VNDetectDocumentSegmentationRequestRevision1`, `VNDetectFaceRectanglesRequestRevision3`, `VNDetectHumanRectanglesRequestRevision2`, `VNGeneratePersonSegmentationRequestRevision1`, `VNRecognizeAnimalsRequestRevision2` | 12.0, 15, 15 |
//! | `VNChirality` `Left`, `Right` | 12.0, 15, 15 |
//! | `VNRecognizeTextRequestRevision3` | 13.0, 16, 16 |
//! | `VNClassifyImageRequestRevision2`, `VNDetectAnimalBodyPoseRequestRevision1`, `VNDetectBarcodesRequestRevision4`, `VNDetectFaceCaptureQualityRequestRevision3`, `VNDetectHumanBodyPose3DRequestRevision1`, `VNGenerateAttentionBasedSaliencyImageRequestRevision2`, `VNGenerateObjectnessBasedSaliencyImageRequestRevision2`, `VNGeneratePersonInstanceMaskRequestRevision1` | 14.0, 17, 17 |
//! | `VNHumanBodyPose3DObservationHeightEstimation` `Reference`, `Measured` | no annotation of its own: its observation's, 14.0, 17, 17 |
//! | `VNCalculateImageAestheticsScoresRequestRevision1` | 15.0, 18, 18 (visionOS 2): the floor |
//!
//! The selectors: every Objective-C message the product code sends to a
//! Vision object or class — each generated method it calls, its one
//! `msg_send!` and the native shims' sends — by the class or protocol that
//! declares it, with the release its header gives it (the declaring class's
//! unless the selector names its own):
//!
//! | declared on | selectors | sent to | introduced |
//! |---|---|---|---|
//! | `NSObject` | `+new` | every request class above, and `VNSequenceRequestHandler` (`ffi.rs`) | the receiving class's |
//! | `VNRequest` | `results` | every request class, each redeclaring it typed | the receiving class's |
//! | `VNRequest` | `revision`, `setRevision:` | the request classes | 10.14, 12, 12 |
//! | `VNRequest` | `+supportedRevisions` | `VNRecognizeTextRequest`, by `msg_send!` (`text.rs`) | 10.14, 12, 12 |
//! | `VNSequenceRequestHandler` | `performRequests:onCGImage:error:`, `performRequests:onImageData:error:` | `VNSequenceRequestHandler` | 10.13, 11, 11 |
//! | the protocol `VNFaceObservationAccepting` | `setInputFaceObservations:` | `VNDetectFaceCaptureQualityRequest`, `VNDetectFaceLandmarksRequest` (`face.rs`) | 10.13, 11, 11 |
//! | `VNObservation` | `confidence` | every observation read | 10.13, 11, 11 |
//! | `VNObservation` | `uuid` | `VNFaceObservation` | 10.13, 11, 11 |
//! | `VNDetectedObjectObservation` | `boundingBox` | its subclasses read | 10.13, 11, 11 |
//! | `VNFaceObservation` | `landmarks` | `VNFaceObservation` | 10.13, 11, 11 |
//! | `VNFaceObservation` | `roll`, `yaw` | `VNFaceObservation` | 10.14, 12, 12 |
//! | `VNFaceObservation` | `faceCaptureQuality` | `VNFaceObservation` | 10.15, 13, 13 |
//! | `VNFaceObservation` | `pitch` | `VNFaceObservation` | 12.0, 15, 15 |
//! | `VNFaceLandmarks` | `confidence` | `VNFaceLandmarks2D` | 10.13, 11, 11 |
//! | `VNFaceLandmarks2D` | `allPoints`, `faceContour`, `leftEye`, `rightEye`, `leftEyebrow`, `rightEyebrow`, `nose`, `noseCrest`, `medianLine`, `outerLips`, `innerLips`, `leftPupil`, `rightPupil` | `VNFaceLandmarks2D` | 10.13, 11, 11 |
//! | `VNFaceLandmarkRegion` | `pointCount` | `VNFaceLandmarkRegion2D` | 10.13, 11, 11 |
//! | `VNFaceLandmarkRegion2D` | `normalizedPoints` | `VNFaceLandmarkRegion2D` | 10.13, 11, 11 |
//! | `VNClassificationObservation` | `identifier` | `VNClassificationObservation`, a recognized object's labels among them | 10.13, 11, 11 |
//! | `VNBarcodeObservation` | `payloadStringValue`, `symbology` | `VNBarcodeObservation` | 10.13, 11, 11 |
//! | `VNHorizonObservation` | `angle` | `VNHorizonObservation` | 10.13, 11, 11 |
//! | `VNRectangleObservation` | `topLeft`, `topRight`, `bottomLeft`, `bottomRight` | `VNRectangleObservation` and its subclasses read | 10.13, 11, 11 |
//! | `VNPixelBufferObservation` | `pixelBuffer` | `VNPixelBufferObservation`, the person segmentation's | 10.13, 11, 11 |
//! | `VNRecognizedObjectObservation` | `labels` | `VNRecognizedObjectObservation` | 10.14, 12, 12 |
//! | `VNRecognizedTextObservation` | `topCandidates:` | `VNRecognizedTextObservation` | 10.15, 13, 13 |
//! | `VNRecognizedText` | `string`, `confidence` | `VNRecognizedText` | 10.15, 13, 13 |
//! | `VNSaliencyImageObservation` | `salientObjects` | `VNSaliencyImageObservation` | 10.15, 13, 13 |
//! | `VNRecognizeTextRequest` | `setRecognitionLevel:`, `setRecognitionLanguages:`, `setUsesLanguageCorrection:`, `setCustomWords:`, `setMinimumTextHeight:` | `VNRecognizeTextRequest` | 10.15, 13, 13 |
//! | `VNRecognizeTextRequest` | `supportedRecognitionLanguagesAndReturnError:` | `VNRecognizeTextRequest` | 12.0, 15, 15 |
//! | `VNRecognizeTextRequest` | `setAutomaticallyDetectsLanguage:` | `VNRecognizeTextRequest` | 13.0, 16, 16 |
//! | `VNPoint` | `x`, `y` | `VNRecognizedPoint` | 11.0, 14, 14 |
//! | `VNDetectedPoint` | `confidence` | `VNRecognizedPoint` | 11.0, 14, 14 |
//! | `VNHumanBodyPoseObservation`, `VNHumanHandPoseObservation` | `recognizedPointsForJointsGroupName:error:` | the two | 11.0, 14, 14 |
//! | `VNDetectHumanHandPoseRequest` | `setMaximumHandCount:` | `VNDetectHumanHandPoseRequest` | 11.0, 14, 14 |
//! | `VNHumanHandPoseObservation` | `chirality` | `VNHumanHandPoseObservation` | 12.0, 15, 15 |
//! | `VNDetectHumanRectanglesRequest` | `setUpperBodyOnly:` | `VNDetectHumanRectanglesRequest` | 12.0, 15, 15 |
//! | `VNAnimalBodyPoseObservation` | `recognizedPointsForJointsGroupName:error:` | `VNAnimalBodyPoseObservation` | 14.0, 17, 17 |
//! | `VNHumanBodyPose3DObservation` | `recognizedPointsForJointsGroupName:error:`, `bodyHeight`, `heightEstimation` | `VNHumanBodyPose3DObservation` | 14.0, 17, 17 |
//! | `VNPoint3D` | `position` | `VNHumanBodyRecognizedPoint3D`, by the native shim (`objc_simd_shim.m`, `avanalyze_0_8_vn_point3d_position`) | 14.0, 17, 17 |
//! | `VNInstanceMaskObservation` | `allInstances`, `generateMaskForInstances:error:` | `VNInstanceMaskObservation` | 14.0, 17, 17 |
//! | `VNImageAestheticsScoresObservation` | `overallScore`, `isUtility` | `VNImageAestheticsScoresObservation` | 15.0, 18, 18 |
//!
//! The Foundation and CoreFoundation census. Foundation is a strong
//! dependency (the second row of the first table), so nothing below is
//! loaded through Vision; each row is reconciled with a call site in
//! `src/` by source, or with the native shim that sends it. "Unannotated"
//! means the SDK's header gives the class or selector no `API_AVAILABLE`
//! of its own, and so no release after Foundation's first (macOS 10.0,
//! iOS 2): older than every system the floor row names, as `NSUUID`
//! (10.8) is the newest annotated one below.
//!
//! The classes and types:
//!
//! | class or type | how it is reached | introduced |
//! |---|---|---|
//! | `NSString` | an instance Vision returns, and `NSString::from_str` (`text.rs`), allocated by the runtime's `objc_alloc` entry point | unannotated |
//! | `NSArray` | an instance Vision returns, and `NSArray::from_retained_slice` (`ffi.rs`, `face.rs`, `text.rs`) | unannotated |
//! | `NSDictionary` | an instance Vision returns (`recognizedPointsForJointsGroupName:error:`); never created here | unannotated |
//! | `NSData` | `NSData::with_bytes` (`ffi.rs`), the encoded image handed to Vision | unannotated |
//! | `NSError` | an instance Vision returns through its `error:` out-parameter; never created here | unannotated |
//! | `NSIndexSet` | an instance Vision returns (`allInstances`, `+supportedRevisions`), and `NSIndexSet::indexSetWithIndex` (`person_mask.rs`) | unannotated |
//! | `NSNumber` | an instance Vision returns (`roll`, `yaw`, `pitch`, `faceCaptureQuality`); never created here | unannotated |
//! | `NSUUID` | an instance Vision returns (`VNObservation.uuid`); never created here, and its class is not looked up | 10.8, 6, 9 |
//! | `NSException` | the type the barrier's `catch (NSException *)` names (`objc_cxx_barrier.mm`), and the instance in flight; never created by product code | unannotated |
//! | `NSEnumerator` | a type only: the `objc2-foundation` feature is on for its iterators, but no `NSEnumerator` is created and none is sent a message, because the array and dictionary iterators fast-enumerate the collection itself and `objectEnumerator` and `keyEnumerator` are never sent. Its class is not looked up | unannotated |
//! | `NSObject` | the root of every class above and below, and the receiver of `+new` (the Vision table above) | unannotated |
//!
//! The one Foundation symbol the crate imports, strong: `_OBJC_EHTYPE_$_NSException`
//! (the exception type-information record), bound by `catch (NSException *)` at
//! `objc_cxx_barrier.mm:314`, defined in CoreFoundation and resolved through the
//! Foundation link of the first table (`nm -m` of the `dead_strip` test binary
//! reads it as `(from CoreFoundation)`). It is unannotated, older than every system the
//! floor row names, and so loads on any system the process can start on. The
//! test-only archive (`objc_cxx_barrier_test.mm`) also imports
//! `_OBJC_CLASS_$_NSException` and `_OBJC_METACLASS_$_NSException`, strong,
//! to raise the exceptions the barrier tests need; it is not part of the
//! library, and the same release applies. Foundation exports no other
//! symbol this crate binds: `NSNotFound` is an integer constant compiled in
//! by `objc2-foundation` (`NSIntegerMax`), and `NSUTF8StringEncoding` is
//! the integer 4 written in `ffi.rs`.
//!
//! The selectors and calls, by the class or protocol that declares them:
//!
//! | declared on | selectors | sent to, and where | introduced |
//! |---|---|---|---|
//! | `NSString` | `lengthOfBytesUsingEncoding:` | every string read from Vision, bounded by `ffi_nsstring_to_smolstr` (`ffi.rs`), and again inside `objc2`'s UTF-8 read below | unannotated |
//! | `NSString` | `UTF8String` | the same strings, by `objc2`'s string read behind `to_smolstr()` (`ffi.rs`); and the barrier, to the in-flight exception's `name` and `reason` (`objc_cxx_barrier.mm`) | unannotated |
//! | `NSString` | `initWithBytes:length:encoding:` | the string `NSString::from_str` allocates (`text.rs`: languages and custom words) | unannotated |
//! | `NSArray` | `count` | every results array Vision returns (`len`, `is_empty`: `face.rs`, `barcode.rs` and the other extractors) | unannotated |
//! | `NSArray` | `initWithObjects:count:` | the array `NSArray::from_retained_slice` allocates (`ffi.rs`, `face.rs`, `text.rs`) | unannotated |
//! | the protocol `NSFastEnumeration` | `countByEnumeratingWithState:objects:count:` | the arrays Vision returns (`.iter()` in every extractor) and the pose dictionaries (`.keys()`, `ffi.rs`); not an `NSEnumerator` | unannotated |
//! | `NSDictionary` | `count` | the pose dictionary (`dict.len()`, `ffi.rs`) | unannotated |
//! | `NSDictionary` | `objectForKey:` | the pose dictionary (`ffi.rs`) | unannotated |
//! | `NSData` | `initWithBytes:length:` | the data `NSData::with_bytes` allocates (`ffi.rs`) | unannotated |
//! | `NSError` | `localizedDescription` | an error Vision returns (`ffi.rs`, `text.rs`) | unannotated |
//! | `NSIndexSet` | `indexSetWithIndex:` | `NSIndexSet` (class message, `person_mask.rs`) | unannotated |
//! | `NSIndexSet` | `firstIndex`, `indexGreaterThanIndex:` | the instance set Vision returns, and the revisions set (`person_mask.rs`, `text.rs`) | unannotated |
//! | `NSIndexSet` | `containsIndex:` | the revisions set of `VNRecognizeTextRequest` (`text.rs`) | unannotated |
//! | `NSNumber` | `floatValue` | the `NSNumber` Vision returns for a face's `roll`, `yaw`, `pitch` and `faceCaptureQuality` (`face.rs`) | unannotated |
//! | `NSUUID` | `UUIDString` | the `uuid` of a `VNFaceObservation` (`face.rs`) | 10.8, 6, 9 |
//! | `NSException` | `name`, `reason` | the exception in flight, in the barrier's catch handler (`objc_cxx_barrier.mm`) | unannotated |
//!
//! These are the only Foundation messages the crate sends. Three kinds of
//! call are not sends and are not rows: `alloc` is the runtime's
//! `objc_alloc`, retain, release and autorelease are the runtime's
//! `objc_retain`, `objc_release` and `objc_autorelease` (the
//! `Retained` clones and drops), and the class lookups are
//! `objc_getClass`. All three are in libobjc, which is linked strongly.
//!
//! The CoreFoundation functions product code calls, every one imported
//! strong from CoreFoundation by `objc2-core-foundation`'s binding, each
//! unannotated in `CFData.h` and `CFBase.h` (older than macOS 10.12):
//!
//! | function | called by |
//! |---|---|
//! | `CFDataCreate` | `CFData::new`, copying a tight pixel plane (`ffi.rs`) |
//! | `CFDataCreateMutable`, `CFDataAppendBytes`, `CFDataGetLength` | `CFMutableData::new`, `CFMutableData::append_bytes` and `length`, copying a strided pixel plane row by row (`ffi.rs`) |
//! | `CFRelease` | the drop of every `CFRetained` (`CFData`, `CFMutableData`, `CGImage`, `CGColorSpace`, `CGDataProvider`) |
//!
//! `CGPoint` and `CGRect` are plain structures: no symbol is imported for
//! them. CoreGraphics and CoreVideo functions are the other strong imports
//! of the first table's second row, each unannotated or older than
//! macOS 10.12 in its header (`CGImageCreate` 10.0, `CGDataProviderCreateWithCFData` 10.4,
//! `CGColorSpaceCreateDeviceRGB` and `…DeviceGray` 10.0, the `CVPixelBuffer…` functions 10.4).
//!
//! No trait sends a message of its own. The nine entry points implement
//! `Debug` by hand from the type name and the revisions cached at
//! construction, so formatting one never reaches `Retained`'s `Debug`
//! (which sends `-[NSObject description]`); `debug_pure` pins that.
//! None derives or implements `PartialEq`, `Eq`, `Hash`, `Display`,
//! `Default` or `Clone`, so no `isEqual:`, `hash` or `description` is
//! reachable through them. Dropping a `Retained` is a release, a runtime
//! call and not a message send, and the crate's own clones of one are
//! retains of the same kind.
//!
//! The other native shim, the exception barrier (`objc_cxx_barrier.mm`),
//! sends Foundation's `name`, `reason` and `UTF8String` alone, to the
//! exception in flight.
//!
//! No Vision symbol is imported, strong or weak:
//! `no_vision_symbol_is_imported` reads the test binary's own symbol
//! table, which holds every entry point's extraction, with `nm -m`.

#[cfg(target_vendor = "apple")]
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(target_vendor = "apple")]
use crate::{AnalyzeError, AnalyzeErrorKind};

/// Hands `$then!` the crate's floor: each Apple system's oldest version,
/// as `system major minor`.
///
/// This is the only place the floor is written. Raising it means editing
/// this list and nothing else.
macro_rules! floor {
  ($then:ident) => {
    $then! { macos 15 0, ios 18 0, tvos 18 0, visionos 2 0 }
  };
}

/// The floor as the text a refusal names.
macro_rules! floor_text {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    concat!(
      "macOS ",
      $macos,
      ".",
      $macos_minor,
      ", iOS ",
      $ios,
      ".",
      $ios_minor,
      ", tvOS ",
      $tvos,
      ".",
      $tvos_minor,
      " or visionOS ",
      $visionos,
      ".",
      $visionos_minor
    )
  };
}

/// The floor as the literals `objc2::available!` takes.
#[cfg(target_vendor = "apple")]
macro_rules! floor_available {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    objc2::available!(
      macos = $macos.$macos_minor,
      ios = $ios.$ios_minor,
      tvos = $tvos.$tvos_minor,
      visionos = $visionos.$visionos_minor
    )
  };
}

/// The floor as the `(major, minor)` this target's system version is
/// compared against. A system outside the four is never at it.
#[cfg(all(test, target_vendor = "apple"))]
macro_rules! floor_version {
  (
    macos $macos:tt $macos_minor:tt,
    ios $ios:tt $ios_minor:tt,
    tvos $tvos:tt $tvos_minor:tt,
    visionos $visionos:tt $visionos_minor:tt
  ) => {
    if cfg!(target_os = "macos") {
      ($macos, $macos_minor)
    } else if cfg!(target_os = "ios") {
      ($ios, $ios_minor)
    } else if cfg!(target_os = "tvos") {
      ($tvos, $tvos_minor)
    } else if cfg!(target_os = "visionos") {
      ($visionos, $visionos_minor)
    } else {
      (u16::MAX, u16::MAX)
    }
  };
}

/// The oldest systems this crate is built on: macOS 15, iOS 18, tvOS 18
/// and visionOS 2.
///
/// Below them, every entry point's constructor refuses with
/// [`AnalyzeErrorKind::BelowFloor`](crate::AnalyzeErrorKind::BelowFloor).
/// It refuses before it sends a message or resolves a Vision class, and
/// its message names this floor.
///
/// The floor is where the newest Vision class an entry point builds
/// arrives. [`VisionAnalyzer`](crate::VisionAnalyzer) always builds
/// `VNCalculateImageAestheticsScoresRequest`, which
/// `VNCalculateImageAestheticsScoresRequest.h` declares
/// `API_AVAILABLE(macos(15.0), ios(18.0), tvos(18.0), visionos(2.0))`. Every
/// other class the nine constructors build is older. One floor for the
/// crate, rather than one per entry point, means a consumer has one
/// question to ask and the crate has one answer to keep true.
///
/// Below a class's own floor, resolving it panics. One check, made first,
/// therefore makes every message a constructor sends safe to send.
pub const SYSTEM_FLOOR: &str = floor!(floor_text);

/// The floor's comparable version on this target, for the test seam.
#[cfg(all(test, target_vendor = "apple"))]
pub(crate) const FLOOR_VERSION: (u16, u16) = floor!(floor_version);

/// Whether this system is at or above [`SYSTEM_FLOOR`].
///
/// objc2 is asked once (it reads the version as Apple's own availability
/// check does, Mac Catalyst and the simulators included), and the answer
/// is kept for the life of the process. Under the test seam the
/// simulated version is compared against the floor instead, and nothing
/// is kept.
#[cfg(target_vendor = "apple")]
fn system_at_floor() -> bool {
  #[cfg(test)]
  if let Some(version) = seam::simulated() {
    return version >= FLOOR_VERSION;
  }
  /// Not asked yet.
  const UNKNOWN: u8 = 0;
  /// At or above the floor.
  const AT: u8 = 1;
  /// Below it.
  const BELOW: u8 = 2;
  // Relaxed is enough: two threads that race to ask both get the same
  // answer and store the same value.
  static ANSWER: AtomicU8 = AtomicU8::new(UNKNOWN);
  match ANSWER.load(Ordering::Relaxed) {
    AT => true,
    BELOW => false,
    _ => {
      let at = floor!(floor_available);
      ANSWER.store(if at { AT } else { BELOW }, Ordering::Relaxed);
      at
    }
  }
}

/// Refuses `site` below [`SYSTEM_FLOOR`] with
/// [`AnalyzeErrorKind::BelowFloor`], naming the entry point and the
/// floor; at or above it, loads Vision ([`crate::ffi::vision`]) and
/// refuses with [`AnalyzeErrorKind::Unsupported`] where it cannot be.
///
/// It is the first statement of every entry point's constructor, before
/// options are checked and before any class is resolved.
#[cfg(target_vendor = "apple")]
pub(crate) fn require_floor(site: &'static str) -> Result<(), AnalyzeError> {
  if !system_at_floor() {
    return Err(AnalyzeError::new(
      AnalyzeErrorKind::BelowFloor,
      format!("{site}: this system is older than {SYSTEM_FLOOR}, the oldest avanalyze is built on"),
    ));
  }
  if crate::ffi::vision().is_none() {
    return Err(AnalyzeError::new(
      AnalyzeErrorKind::Unsupported,
      format!("{site}: Vision.framework could not be loaded"),
    ));
  }
  Ok(())
}

/// The test seam: a simulated system version for the calling thread.
///
/// It is thread-local, so tests that run side by side never see each
/// other's simulation. It is scoped, so a simulation ends when its guard
/// drops, a panicking test's included.
#[cfg(all(test, target_vendor = "apple"))]
pub(crate) mod seam {
  use std::cell::Cell;

  thread_local! {
    static SIMULATED: Cell<Option<(u16, u16)>> = const { Cell::new(None) };
  }

  /// The calling thread's simulated `(major, minor)`, if any.
  pub(crate) fn simulated() -> Option<(u16, u16)> {
    SIMULATED.with(Cell::get)
  }

  /// A simulation in force. Dropping it restores what was simulated
  /// before.
  #[must_use = "the simulation ends when the guard drops"]
  pub(crate) struct Simulated {
    previous: Option<(u16, u16)>,
  }

  /// Simulates `major.minor` as the calling thread's system version until
  /// the guard drops.
  pub(crate) fn simulate(major: u16, minor: u16) -> Simulated {
    Simulated {
      previous: SIMULATED.with(|cell| cell.replace(Some((major, minor)))),
    }
  }

  impl Drop for Simulated {
    fn drop(&mut self) {
      SIMULATED.with(|cell| cell.set(self.previous));
    }
  }
}
