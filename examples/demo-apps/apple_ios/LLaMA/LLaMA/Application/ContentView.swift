/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

import SwiftUI
import UniformTypeIdentifiers

import LLaMARunner

class RunnerHolder: ObservableObject {
  var llamaRunner: LLaMARunner?
  var llavaRunner: LLaVARunner?
}

extension UIImage {
  func resized(to newSize: CGSize) -> UIImage {
    let format = UIGraphicsImageRendererFormat.default()
    format.scale = 1
    return UIGraphicsImageRenderer(size: newSize, format: format).image {
      _ in draw(in: CGRect(origin: .zero, size: newSize))
    }
  }

  func toRGBArray() -> [UInt8]? {
    guard let cgImage = self.cgImage else { return nil }

    let width = Int(cgImage.width), height = Int(cgImage.height)
    let totalPixels = width * height, bytesPerPixel = 4, bytesPerRow = bytesPerPixel * width
    var rgbValues = [UInt8](repeating: 0, count: totalPixels * 3)
    var pixelData = [UInt8](repeating: 0, count: width * height * bytesPerPixel)

    guard let context = CGContext(
      data: &pixelData, width: width, height: height, bitsPerComponent: 8,
      bytesPerRow: bytesPerRow, space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue
    ) else { return nil }

    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    for y in 0..<height {
      for x in 0..<width {
        let pixelIndex = (y * width + x) * bytesPerPixel
        let rgbIndex = y * width + x
        rgbValues[rgbIndex] = pixelData[pixelIndex]
        rgbValues[rgbIndex + totalPixels] = pixelData[pixelIndex + 1]
        rgbValues[rgbIndex + totalPixels * 2] = pixelData[pixelIndex + 2]
      }
    }
    return rgbValues
  }
}

struct ContentView: View {
  @State private var prompt = ""
  @State private var messages: [Message] = []
  @State private var showingLogs = false
  @State private var pickerType: PickerType?
  @State private var isGenerating = false
  @State private var shouldStopGenerating = false
  @State private var shouldStopShowingToken = false
  @State private var thinkingMode = false
  @State private var showThinkingModeNotification = false
  private let runnerQueue = DispatchQueue(label: "org.pytorch.executorch.llama")
  @StateObject private var runnerHolder = RunnerHolder()
  @StateObject private var resourceManager = ResourceManager()
  @StateObject private var resourceMonitor = ResourceMonitor()
  @StateObject private var logManager = LogManager()

  @State private var isImagePickerPresented = false
  @State private var selectedImage: UIImage?
  @State private var imagePickerSourceType: UIImagePickerController.SourceType = .photoLibrary

  @State private var showingSettings = false
  @FocusState private var textFieldFocused: Bool

  enum PickerType {
    case model
    case tokenizer
  }

  enum ModelType {
    case llama
    case llava
    case qwen3
    case phi4

    static func fromPath(_ path: String) -> ModelType {
      let filename = (path as NSString).lastPathComponent.lowercased()
      if filename.hasPrefix("llama") {
        return .llama
      } else if filename.hasPrefix("llava") {
        return .llava
      } else if filename.hasPrefix("qwen3") {
        return .qwen3
      } else if filename.hasPrefix("phi4") {
        return .phi4
      }
      print("Unknown model type in path: \(path). Model filename should start with one of: llama, llava, qwen3, or phi4")
      exit(1)
    }
  }

  private var placeholder: String {
    resourceManager.isModelValid ? resourceManager.isTokenizerValid ? "Prompt..." : "Select Tokenizer..." : "Select Model..."
  }

  private var title: String {
    resourceManager.isModelValid ? resourceManager.isTokenizerValid ? resourceManager.modelName : "Select Tokenizer..." : "Select Model..."
  }

  private var modelTitle: String {
    resourceManager.isModelValid ? resourceManager.modelName : "Select Model..."
  }

  private var tokenizerTitle: String {
    resourceManager.isTokenizerValid ? resourceManager.tokenizerName : "Select Tokenizer..."
  }

  private var isInputEnabled: Bool { resourceManager.isModelValid && resourceManager.isTokenizerValid }

  var body: some View {
    NavigationView {
      ZStack {
        VStack {
          if showingSettings {
            VStack(spacing: 20) {
              HStack {
                VStack(spacing: 10) {
                  Button(action: { pickerType = .model }) {
                    Label(modelTitle, systemImage: "doc")
                      .lineLimit(1)
                      .truncationMode(.middle)
                      .frame(maxWidth: 300, alignment: .leading)
                  }
                  Button(action: { pickerType = .tokenizer }) {
                    Label(tokenizerTitle, systemImage: "doc")
                      .lineLimit(1)
                      .truncationMode(.middle)
                      .frame(maxWidth: 300, alignment: .leading)
                  }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
                .fixedSize(horizontal: true, vertical: false)
                Spacer()
              }
              .padding()
            }
          }

          MessageListView(messages: $messages)
            .simultaneousGesture(
              DragGesture().onChanged { value in
                if value.translation.height > 10 {
                  hideKeyboard()
                }
                showingSettings = false
                textFieldFocused = false
              }
            )
            .onTapGesture {
              showingSettings = false
              textFieldFocused = false
            }

          HStack {
            Button(action: {
              imagePickerSourceType = .photoLibrary
              isImagePickerPresented = true
            }) {
              Image(systemName: "photo.on.rectangle")
                .resizable()
                .scaledToFit()
                .frame(width: 24, height: 24)
            }
            .background(Color.clear)
            .cornerRadius(8)

            Button(action: {
              if UIImagePickerController.isSourceTypeAvailable(.camera) {
                imagePickerSourceType = .camera
                isImagePickerPresented = true
              } else {
                print("Camera not available")
              }
            }) {
              Image(systemName: "camera")
                .resizable()
                .scaledToFit()
                .frame(width: 24, height: 24)
            }
            .background(Color.clear)
            .cornerRadius(8)

            if resourceManager.isModelValid && ModelType.fromPath(resourceManager.modelPath) == .qwen3 {
              Button(action: {
                thinkingMode.toggle()
                showThinkingModeNotification = true
                DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                  showThinkingModeNotification = false
                }
              }) {
                Image(systemName: "brain")
                  .resizable()
                  .scaledToFit()
                  .frame(width: 24, height: 24)
                  .foregroundColor(thinkingMode ? .blue : .gray)
              }
              .background(Color.clear)
              .cornerRadius(8)
            }

            TextField(placeholder, text: $prompt, axis: .vertical)
              .padding(8)
              .background(Color.gray.opacity(0.1))
              .cornerRadius(20)
              .lineLimit(1...10)
              .overlay(
                RoundedRectangle(cornerRadius: 20)
                  .stroke(isInputEnabled ? Color.blue : Color.gray, lineWidth: 1)
              )
              .disabled(!isInputEnabled)
              .focused($textFieldFocused)
              .onAppear { textFieldFocused = false }
              .onTapGesture {
                showingSettings = false
              }

            Button(action: isGenerating ? stop : generate) {
              Image(systemName: isGenerating ? "stop.circle" : "arrowshape.up.circle.fill")
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(height: 28)
            }
            .disabled(isGenerating ? shouldStopGenerating : (!isInputEnabled || prompt.isEmpty))
          }
          .padding([.leading, .trailing, .bottom], 10)
        }
        .sheet(isPresented: $isImagePickerPresented, onDismiss: addSelectedImageMessage) {
          ImagePicker(selectedImage: $selectedImage, sourceType: imagePickerSourceType)
            .id(imagePickerSourceType.rawValue)
        }

        if showThinkingModeNotification {
          Text(thinkingMode ? "Thinking mode enabled" : "Thinking mode disabled")
            .padding(8)
            .background(Color(UIColor.secondarySystemBackground))
            .cornerRadius(8)
            .transition(.opacity)
            .animation(.easeInOut(duration: 0.2), value: showThinkingModeNotification)
        }
      }
      .navigationBarTitle(title, displayMode: .inline)
      .navigationBarItems(
        leading:
          Button(action: {
            showingSettings.toggle()
          }) {
            Image(systemName: "folder")
              .imageScale(.large)
          },
        trailing:
          HStack {
            Menu {
              Section(header: Text("Memory")) {
                Text("Used: \(resourceMonitor.usedMemory) Mb")
                Text("Available: \(resourceMonitor.usedMemory) Mb")
              }
            } label: {
              Text("\(resourceMonitor.usedMemory) Mb")
            }
            .onAppear {
              resourceMonitor.start()
            }
            .onDisappear {
              resourceMonitor.stop()
            }
            Button(action: { showingLogs = true }) {
              Image(systemName: "list.bullet.rectangle")
            }
          }
      )
      .sheet(isPresented: $showingLogs) {
        NavigationView {
          LogView(logManager: logManager)
        }
      }
      .fileImporter(
        isPresented: Binding<Bool>(
          get: { pickerType != nil },
          set: { if !$0 { pickerType = nil } }
        ),
        allowedContentTypes: allowedContentTypes(),
        allowsMultipleSelection: false
      ) { [pickerType] result in
        handleFileImportResult(pickerType, result)
      }
      .onAppear {
        do {
          try resourceManager.createDirectoriesIfNeeded()
        } catch {
          withAnimation {
            messages.append(Message(type: .info, text: "Error creating content directories: \(error.localizedDescription)"))
          }
        }
      }
    }
    .navigationViewStyle(StackNavigationViewStyle())
  }

  private func addSelectedImageMessage() {
    if let selectedImage {
      messages.append(Message(image: selectedImage))
    }
  }

  private func generate() {
    guard !prompt.isEmpty else { return }
    isGenerating = true
    shouldStopGenerating = false
    shouldStopShowingToken = false
    let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    let seq_len = 768 // text: 256, vision: 768
    let modelPath = resourceManager.modelPath
    let tokenizerPath = resourceManager.tokenizerPath
    let modelType = ModelType.fromPath(modelPath)

    prompt = ""
    hideKeyboard()
    showingSettings = false

    messages.append(Message(text: text))
    messages.append(Message(type: modelType == .llama ? .llamagenerated : .llavagenerated))

    runnerQueue.async {
      defer {
        DispatchQueue.main.async {
          isGenerating = false
          selectedImage = nil
        }
      }

      switch modelType {
      case .llama, .qwen3, .phi4:
        runnerHolder.llamaRunner = runnerHolder.llamaRunner ?? LLaMARunner(modelPath: modelPath, tokenizerPath: tokenizerPath)
      case .llava:
        runnerHolder.llavaRunner = runnerHolder.llavaRunner ?? LLaVARunner(modelPath: modelPath, tokenizerPath: tokenizerPath)
      }

      guard !shouldStopGenerating else { return }
      switch modelType {
      case .llama, .qwen3, .phi4:
        if let runner = runnerHolder.llamaRunner, !runner.isLoaded() {
          var error: Error?
          let startLoadTime = Date()
          do {
            try runner.load()
          } catch let loadError {
            error = loadError
          }

          let loadTime = Date().timeIntervalSince(startLoadTime)
          DispatchQueue.main.async {
            withAnimation {
              var message = messages.removeLast()
              message.type = .info
              if let error {
                message.text = "Model loading failed: error \((error as NSError).code)"
              } else {
                message.text = "Model loaded in \(String(format: "%.2f", loadTime)) s"
              }
              messages.append(message)
              if error == nil {
                messages.append(Message(type: .llamagenerated))
              }
            }
          }
          if error != nil {
            return
          }
        }
      case .llava:
        if let runner = runnerHolder.llavaRunner, !runner.isLoaded() {
          var error: Error?
          let startLoadTime = Date()
          do {
            try runner.load()
          } catch let loadError {
            error = loadError
          }

          let loadTime = Date().timeIntervalSince(startLoadTime)
          DispatchQueue.main.async {
            withAnimation {
              var message = messages.removeLast()
              message.type = .info
              if let error {
                message.text = "Model loading failed: error \((error as NSError).code)"
              } else {
                message.text = "Model loaded in \(String(format: "%.2f", loadTime)) s"
              }
              messages.append(message)
              if error == nil {
                messages.append(Message(type: .llavagenerated))
              }
            }
          }
          if error != nil {
            return
          }
        }
      }

      guard !shouldStopGenerating else {
        DispatchQueue.main.async {
          withAnimation {
            _ = messages.removeLast()
          }
        }
        return
      }
      do {
        var tokens: [String] = []
        var rgbArray: [UInt8]?
        let MAX_WIDTH = 336.0
        var newHeight = 0.0
        var imageBuffer: UnsafeMutableRawPointer?

        if let img = selectedImage {
          let llava_prompt = "\(text) ASSISTANT"

          newHeight = MAX_WIDTH * img.size.height / img.size.width
          let resizedImage = img.resized(to: CGSize(width: MAX_WIDTH, height: newHeight))
          rgbArray = resizedImage.toRGBArray()
          imageBuffer = UnsafeMutableRawPointer(mutating: rgbArray)

          try runnerHolder.llavaRunner?.generate(imageBuffer!, width: MAX_WIDTH, height: newHeight, prompt: llava_prompt, sequenceLength: seq_len) { token in

            if token != llava_prompt {
              if token == "</s>" {
                shouldStopGenerating = true
                runnerHolder.llavaRunner?.stop()
              } else {
                tokens.append(token)
                if tokens.count > 2 {
                  let text = tokens.joined()
                  let count = tokens.count
                  tokens = []
                  DispatchQueue.main.async {
                    var message = messages.removeLast()
                    message.text += text
                    message.tokenCount += count
                    message.dateUpdated = Date()
                    messages.append(message)
                  }
                }
                if shouldStopGenerating {
                  runnerHolder.llavaRunner?.stop()
                }
              }
            }
          }
        } else {
          let prompt: String
          switch modelType {
          case .qwen3:
            let basePrompt = String(format: Constants.qwen3PromptTemplate, text)
            // If thinking mode is enabled for Qwen, don't skip the <think></think> special tokens
            // and have them be generated.
            prompt = thinkingMode ? basePrompt.replacingOccurrences(of: "<think>\n\n</think>\n\n\n", with: "") : basePrompt
          case .llama:
            prompt = String(format: Constants.llama3PromptTemplate, text)
          case .llava:
            prompt = String(format: Constants.llama3PromptTemplate, text)
          case .phi4:
              prompt = String(format: Constants.phi4PromptTemplate, text)
          }

          try runnerHolder.llamaRunner?.generate(prompt, sequenceLength: seq_len) { token in

            if token != prompt {
                if token == "<|eot_id|>" {
                // hack to fix the issue that extension/llm/runner/text_token_generator.h
                // keeps generating after <|eot_id|>
                shouldStopShowingToken = true
              } else if token == "<|im_end|>" {
                // Qwen3 specific token.
                // Skip.
              } else if token == "<think>" {
                // Qwen3 specific token.
                let textToFlush = tokens.joined()
                let flushedTokenCount = tokens.count
                tokens = []
                DispatchQueue.main.async {
                  var message = messages.removeLast()
                  message.text += textToFlush
                  message.text += message.text.isEmpty ? "Thinking...\n\n" : "\n\nThinking...\n\n"
                  message.tokenCount += flushedTokenCount + 1  // + 1 for the start thinking token.
                  message.dateUpdated = Date()
                  messages.append(message)
                }
              } else if token == "</think>" {
                // Qwen3 specific token.
                let textToFlush = tokens.joined()
                let flushedTokenCount = tokens.count
                tokens = []
                DispatchQueue.main.async {
                  var message = messages.removeLast()
                  message.text += textToFlush
                  message.text += "\n\nFinished thinking.\n\n"
                  message.tokenCount += flushedTokenCount + 1  // + 1 for the end thinking token.
                  message.dateUpdated = Date()
                  messages.append(message)
                }
              } else {
                tokens.append(token.trimmingCharacters(in: .newlines))
                // Flush tokens in groups of 3 so that it's closer to whole words being generated
                // rather than parts of words (tokens).
                if tokens.count > 2 {
                  let text = tokens.joined()
                  let count = tokens.count
                  tokens = []
                  DispatchQueue.main.async {
                    var message = messages.removeLast()
                    message.text += text
                    message.tokenCount += count
                    message.dateUpdated = Date()
                    messages.append(message)
                  }
                }
                if shouldStopGenerating {
                  runnerHolder.llamaRunner?.stop()
                }
              }
            }
          }
        }
      } catch {
        DispatchQueue.main.async {
          withAnimation {
            var message = messages.removeLast()
            message.type = .info
            message.text = "Text generation failed: error \((error as NSError).code)"
            messages.append(message)
          }
        }
      }
    }
  }

  private func stop() {
    shouldStopGenerating = true
  }

  private func allowedContentTypes() -> [UTType] {
    guard let pickerType else { return [] }
    switch pickerType {
    case .model:
      return [UTType(filenameExtension: "pte")].compactMap { $0 }
    case .tokenizer:
      return [UTType(filenameExtension: "bin"), UTType(filenameExtension: "model"), UTType(filenameExtension: "json"), ].compactMap { $0 }
    }
  }

  private func handleFileImportResult(_ pickerType: PickerType?, _ result: Result<[URL], Error>) {
    switch result {
    case .success(let urls):
      guard let url = urls.first, let pickerType else {
        withAnimation {
          messages.append(Message(type: .info, text: "Failed to select a file"))
        }
        return
      }
      runnerQueue.async {
        runnerHolder.llamaRunner = nil
        runnerHolder.llavaRunner = nil
      }
      switch pickerType {
      case .model:
        resourceManager.modelPath = url.path
      case .tokenizer:
        resourceManager.tokenizerPath = url.path
      }
      if resourceManager.isModelValid && resourceManager.isTokenizerValid {
        showingSettings = false
        textFieldFocused = true
      }
    case .failure(let error):
      withAnimation {
        messages.append(Message(type: .info, text: "Failed to select a file: \(error.localizedDescription)"))
      }
    }
  }
}

extension View {
  func hideKeyboard() {
    UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
  }
}
