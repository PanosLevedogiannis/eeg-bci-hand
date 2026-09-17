// H.264 + AAC at a chosen bitrate, with AVFoundation only.
//
//   swift site/transcode.swift SRC OUT.mp4 WIDTH HEIGHT VIDEO_KBPS [START DURATION]
//
// avconvert's presets pick their own bitrate, which for a phone clip comes to
// about 7 Mbit/s at 720p: far too much for a page that carries the clip inside
// itself. prepare_video.py calls this instead when --kbps is given.
import AVFoundation

let a = CommandLine.arguments
guard a.count >= 6 else {
    FileHandle.standardError.write("usage: SRC OUT WIDTH HEIGHT VIDEO_KBPS [START DURATION]\n".data(using: .utf8)!)
    exit(2)
}
let src = URL(fileURLWithPath: a[1]), dst = URL(fileURLWithPath: a[2])
let W = Int(a[3])!, H = Int(a[4])!, kbps = Int(a[5])!
let start = a.count > 6 ? Double(a[6])! : 0
let asset = AVURLAsset(url: src)
let total = CMTimeGetSeconds(asset.duration)
let length = a.count > 7 ? min(Double(a[7])!, total - start) : total - start

func fail(_ m: String) -> Never {
    FileHandle.standardError.write((m + "\n").data(using: .utf8)!)
    exit(1)
}

guard let vTrack = asset.tracks(withMediaType: .video).first else { fail("no video track") }
let aTrack = asset.tracks(withMediaType: .audio).first

try? FileManager.default.removeItem(at: dst)
let reader = try! AVAssetReader(asset: asset)
reader.timeRange = CMTimeRange(start: CMTime(seconds: start, preferredTimescale: 600),
                               duration: CMTime(seconds: length, preferredTimescale: 600))
let writer = try! AVAssetWriter(outputURL: dst, fileType: .mp4)
writer.shouldOptimizeForNetworkUse = true          // moov first: the page can start playing early

// size the stored frames as the source stores them; a phone's rotation is
// carried over below as metadata, as the source has it
let nat = vTrack.naturalSize
let outW = nat.height > nat.width ? H : W, outH = nat.height > nat.width ? W : H

let vOut = AVAssetReaderTrackOutput(track: vTrack, outputSettings: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange])
vOut.alwaysCopiesSampleData = false
reader.add(vOut)
let vIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: outW, AVVideoHeightKey: outH,
    AVVideoScalingModeKey: AVVideoScalingModeResizeAspect,
    AVVideoCompressionPropertiesKey: [
        AVVideoAverageBitRateKey: kbps * 1000,
        AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
        AVVideoMaxKeyFrameIntervalDurationKey: 2,
        AVVideoAllowFrameReorderingKey: true,
    ]])
vIn.expectsMediaDataInRealTime = false
vIn.transform = vTrack.preferredTransform
writer.add(vIn)

var pairs: [(AVAssetReaderOutput, AVAssetWriterInput)] = [(vOut, vIn)]
if let t = aTrack {
    let aOut = AVAssetReaderTrackOutput(track: t, outputSettings: [
        AVFormatIDKey: kAudioFormatLinearPCM, AVNumberOfChannelsKey: 2, AVSampleRateKey: 44100,
        AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
        AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false])
    let aIn = AVAssetWriterInput(mediaType: .audio, outputSettings: [
        AVFormatIDKey: kAudioFormatMPEG4AAC, AVNumberOfChannelsKey: 2,
        AVSampleRateKey: 44100, AVEncoderBitRateKey: 96000])
    aIn.expectsMediaDataInRealTime = false
    if reader.canAdd(aOut) && writer.canAdd(aIn) {
        reader.add(aOut); writer.add(aIn); pairs.append((aOut, aIn))
    } else {
        print("audio track cannot be converted, writing video only")
    }
}

guard reader.startReading() else { fail("reader: \(String(describing: reader.error))") }
guard writer.startWriting() else { fail("writer: \(String(describing: writer.error))") }
writer.startSession(atSourceTime: reader.timeRange.start)

let group = DispatchGroup()
for (i, (out, inp)) in pairs.enumerated() {
    group.enter()
    inp.requestMediaDataWhenReady(on: DispatchQueue(label: "pump\(i)")) {
        while inp.isReadyForMoreMediaData {
            if let buf = out.copyNextSampleBuffer() {
                if !inp.append(buf) { inp.markAsFinished(); group.leave(); return }
            } else {
                inp.markAsFinished(); group.leave(); return
            }
        }
    }
}
group.wait()

let done = DispatchSemaphore(value: 0)
writer.finishWriting { done.signal() }
done.wait()
if writer.status != .completed { fail("failed: \(String(describing: writer.error))") }
print(String(format: "%dx%d, %d kbit/s, %.1f s", outW, outH, kbps, length))
